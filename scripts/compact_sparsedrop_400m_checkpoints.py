#!/usr/bin/env python3
"""Retain verified inference weights only after a 400M run completes 20B tokens.

Both replacement files are saved, reloaded and compared tensor-for-tensor
before either original checkpoint can be removed. A durable retention report
supports recovery after an interruption between the two removals. This command
does not accept a smaller model, a different budget, or an in-progress run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


TOKEN_BUDGET = 20_000_000_000
RETENTION_FILE = "checkpoint_retention.json"
EXCLUDED_STATE = {"optimizer", "sampler", "rng"}
EXECUTION_KEYS = {"batch_size", "grad_accum", "gradient_checkpointing"}


@dataclass(frozen=True)
class _ModelContract:
    vocab: int = 49152
    hidden: int = 2048
    layers: int = 6
    heads: int = 32
    intermediate: int = 5504
    parameters: int = 404_279_297


_PRODUCTION = _ModelContract()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
            "inode": value.st_ino, "device": value.st_dev}


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_weights(path: Path, value: dict[str, Any], torch: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_run(summary: dict[str, Any], config: dict[str, Any], contract: _ModelContract) -> None:
    _require(summary.get("status") == "complete", "Run must have status=complete")
    _require(type(summary.get("tokens_seen")) is int and summary["tokens_seen"] == TOKEN_BUDGET,
             "Run must have completed exactly 20B tokens")
    _require(summary.get("token_budget_completed") is True, "The 20B token budget is not completed")
    _require(summary.get("completion_reason") == "original_budget", "An early-stop run cannot be compacted")
    _require(summary.get("actual_tokens_seen") == TOKEN_BUDGET
             and summary.get("effective_target_tokens") == TOKEN_BUDGET
             and summary.get("original_token_budget") == TOKEN_BUDGET,
             "Completion token counts disagree")
    _require(summary.get("overshoot_steps") == 0 and summary.get("overshoot_tokens") == 0,
             "An overbudget run cannot be compacted")
    _require(config.get("parameters_total") == contract.parameters,
             "Model parameter count differs from the required 404M model")
    args, model = config.get("arguments", {}), config.get("model", {})
    dimensions = {"vocab_size": contract.vocab, "hidden_size": contract.hidden,
                  "num_layers": contract.layers, "num_heads": contract.heads,
                  "intermediate_size": contract.intermediate}
    for key, expected in dimensions.items():
        _require(args.get(key) == expected and model.get(key) == expected,
                 f"Model configuration mismatch: {key}")
    _require(args.get("loops") in {2, 4, 6, 8, 10, 12}, "Unexpected loop count")
    _require(args.get("mode") in {"ffn", "attention", "both"}, "Unexpected dropout setting")
    for key, expected in {"p": 0.1, "schedule": "constant", "delta": 0.0, "seed": 42,
                          "token_budget": TOKEN_BUDGET, "seq_len": 512}.items():
        _require(args.get(key) == expected, f"Experiment configuration mismatch: {key}")
    _require(_valid_execution_profile({key: args.get(key) for key in EXECUTION_KEYS}),
             "Invalid execution profile or effective optimizer batch")
    _require(args.get("stop_after_steps") is None, "A stopping cap is still set")
    model_settings = {"max_recurrent_steps": args["loops"], "max_seq_len": args["seq_len"],
                      "residual_dropout_mode": args["mode"], "residual_dropout_p": 0.1,
                      "residual_dropout_schedule": "constant", "residual_dropout_delta": 0.0,
                      "dropout": 0.0, "beta_kl": 0.0}
    for key, expected in model_settings.items():
        _require(model.get(key) == expected, f"Model dropout configuration mismatch: {key}")
    capacity = args["batch_size"] * args["grad_accum"] * args["seq_len"]
    steps = (TOKEN_BUDGET + capacity - 1) // capacity
    _require(summary.get("step") == steps and summary.get("actual_steps") == steps
             and summary.get("effective_target_steps") == steps and args.get("max_steps") == steps,
             "Final optimizer-step counts disagree with the token budget")
    _require(config.get("status") == "complete" and config.get("token_budget_completed") is True,
             "Run configuration is not marked complete")


def _expected_shapes(contract: _ModelContract) -> dict[str, tuple[int, ...]]:
    d, m, v = contract.hidden, contract.intermediate, contract.vocab
    shapes = {"embed.weight": (v, d), "lm_head.weight": (v, d), "final_norm.weight": (d,),
              "exit_gate.weight": (1, d), "exit_gate.bias": (1,)}
    for layer in range(contract.layers):
        prefix = f"layers.{layer}."
        for name in ("pre_attn_norm", "post_attn_norm", "pre_ffn_norm", "post_ffn_norm"):
            shapes[prefix + name + ".weight"] = (d,)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[prefix + "attn." + name + ".weight"] = (d, d)
        for name in ("w_gate", "w_up"):
            shapes[prefix + "ffn." + name + ".weight"] = (m, d)
        shapes[prefix + "ffn.w_down.weight"] = (d, m)
    return shapes


def _valid_execution_profile(profile: Any) -> bool:
    return (isinstance(profile, dict) and set(profile) == EXECUTION_KEYS
            and type(profile.get("batch_size")) is int and profile["batch_size"] > 0
            and type(profile.get("grad_accum")) is int and profile["grad_accum"] > 0
            and profile["batch_size"] * profile["grad_accum"] == 32
            and type(profile.get("gradient_checkpointing")) is bool)


def _historical_best_profile(original: dict[str, Any], step: int,
                             config: dict[str, Any]) -> bool:
    """Follow recorded transitions after an earlier best checkpoint, without relabeling it."""
    profile = {key: original.get(key) for key in EXECUTION_KEYS}
    current = {key: config["arguments"].get(key) for key in EXECUTION_KEYS}
    history = config.get("execution_history", [])
    if not _valid_execution_profile(profile) or not isinstance(history, list):
        return False
    applied, previous_step = False, -1
    for transition in history:
        if not isinstance(transition, dict):
            return False
        boundary = transition.get("step")
        old, new = transition.get("from"), transition.get("to")
        if (type(boundary) is not int or boundary < previous_step
                or not _valid_execution_profile(old) or not _valid_execution_profile(new)
                or transition.get("tokens_seen") != min(boundary * 16384, TOKEN_BUDGET)):
            return False
        previous_step = boundary
        if boundary < step:
            continue
        if profile != old:
            return False
        profile, applied = new, True
    return applied and profile == current


def _validate_checkpoint(checkpoint: dict[str, Any], role: str, summary: dict[str, Any],
                         config: dict[str, Any], contract: _ModelContract, torch: Any,
                         *, compact: bool) -> None:
    _require(isinstance(checkpoint, dict), "Checkpoint must be a mapping")
    _require(checkpoint.get("model_config") == config["model"], "Checkpoint model configuration differs")
    ignored = {"config", "output_dir", "resume", "log_every", "checkpoint_every", "max_seconds"}
    original_args = checkpoint.get("arguments", {})
    _require(isinstance(original_args, dict), "Checkpoint arguments are missing")
    step = checkpoint.get("step")
    _require(type(step) is int and 0 < step <= summary["step"], "Checkpoint step is invalid")
    historical_best = (role == "best" and _historical_best_profile(original_args, step, config))
    for key, value in config["arguments"].items():
        if key not in ignored:
            _require(original_args.get(key) == value or (historical_best and key in EXECUTION_KEYS),
                     f"Checkpoint arguments differ: {key}")
    capacity = config["arguments"]["batch_size"] * config["arguments"]["grad_accum"] * config["arguments"]["seq_len"]
    _require(checkpoint.get("tokens_seen") == min(step * capacity, TOKEN_BUDGET),
             "Checkpoint token count is inconsistent")
    if role == "last":
        _require(step == summary["step"] and checkpoint["tokens_seen"] == TOKEN_BUDGET,
                 "Last checkpoint has not reached the completed budget")
        _require(checkpoint.get("execution_history", []) == config.get("execution_history", []),
                 "Last checkpoint execution history differs from the run metadata")
    _require(checkpoint.get("initial_parameters_sha256") == config.get("initial_parameters_sha256")
             and isinstance(config.get("initial_parameters_sha256"), str),
             "Checkpoint initialization identity differs")
    tensors = checkpoint.get("model")
    shapes = _expected_shapes(contract)
    _require(isinstance(tensors, dict) and set(tensors) == set(shapes),
             "Model state keys differ from the expected dense model")
    for name, shape in shapes.items():
        tensor = tensors[name]
        _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape
                 and tensor.dtype == torch.float32, f"Model tensor shape/dtype mismatch: {name}")
        _require(bool(torch.isfinite(tensor).all()), f"Nonfinite model weights: {name}")
    _require(torch.equal(tensors["embed.weight"], tensors["lm_head.weight"]),
             "Tied embedding and output weights differ")
    count = sum(t.numel() for key, t in tensors.items() if key != "lm_head.weight")
    _require(count == contract.parameters, "Actual model tensors have the wrong parameter count")
    if compact:
        _require(checkpoint.get("weights_only") is True and checkpoint.get("resumable") is False,
                 "Replacement must be explicitly non-resumable weights only")
        _require(not EXCLUDED_STATE.intersection(checkpoint), "Replacement still contains training state")
        _require(checkpoint.get("metadata") == config, "Replacement run metadata differs")
    else:
        _require(all(name in checkpoint for name in EXCLUDED_STATE),
                 "Original checkpoint is missing optimizer, sampler or RNG state")


def _verify_equal(source: dict[str, Any], replacement: dict[str, Any], torch: Any) -> None:
    _require(set(source["model"]) == set(replacement["model"]), "Replacement model keys differ")
    for name, tensor in source["model"].items():
        other = replacement["model"][name]
        _require(tensor.dtype == other.dtype and tensor.shape == other.shape
                 and torch.equal(tensor, other), f"Replacement changed model tensor: {name}")
    for key, value in source.items():
        if key not in EXCLUDED_STATE and key != "model":
            _require(replacement.get(key) == value, f"Replacement changed checkpoint metadata: {key}")


def _compact_locked(output: Path, contract: _ModelContract) -> dict[str, Any]:
    import torch

    summary_path, config_path = output / "summary.json", output / "config.json"
    summary, config = _read_json(summary_path), _read_json(config_path)
    _validate_run(summary, config, contract)
    config_digest = _sha256(config_path)
    summary_digest = _sha256(summary_path)
    report_path = output / RETENTION_FILE
    previous = _read_json(report_path) if report_path.exists() else None
    if previous is not None:
        _require(previous.get("schema_version") == "sparsedrop_400m_retention_v1"
                 and previous.get("config_sha256") == config_digest,
                 "Existing retention report belongs to a different run/configuration")
    records: dict[str, Any] = {}
    source_identities: dict[str, Any] = {}
    for role in ("best", "last"):
        source_path, target_path = output / f"{role}.pt", output / f"{role}_model.pt"
        _require(not source_path.is_symlink() and not target_path.is_symlink(),
                 "Checkpoint symlinks are not supported")
        if source_path.exists():
            identity = _identity(source_path)
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            _validate_checkpoint(source, role, summary, config, contract, torch, compact=False)
            if not target_path.exists():
                replacement = {key: value for key, value in source.items() if key not in EXCLUDED_STATE}
                replacement.update(weights_only=True, resumable=False, metadata=deepcopy(config))
                _save_weights(target_path, replacement, torch)
                del replacement
            replacement = torch.load(target_path, map_location="cpu", weights_only=False)
            _validate_checkpoint(replacement, role, summary, config, contract, torch, compact=True)
            _verify_equal(source, replacement, torch)
            _require(_identity(source_path) == identity, "Original checkpoint changed during verification")
            source_identities[role] = identity
            records[role] = {
                "source": source_path.name, "replacement": target_path.name,
                "source_identity": identity, "replacement_bytes": target_path.stat().st_size,
                "replacement_sha256": _sha256(target_path),
                "step": source["step"], "tokens_seen": source["tokens_seen"],
                "model_tensor_count": len(source["model"]),
                "all_tensor_keys_dtypes_shapes_values_equal": True,
            }
            del source, replacement
        else:
            _require(previous is not None and previous.get("status") in {"verified", "complete"},
                     f"Missing {source_path.name} without a prior verified retention report")
            record = previous.get("checkpoints", {}).get(role, {})
            _require(record.get("replacement") == target_path.name
                     and record.get("all_tensor_keys_dtypes_shapes_values_equal") is True,
                     f"No prior tensor equality proof for {role}")
            _require(target_path.is_file() and _sha256(target_path) == record.get("replacement_sha256"),
                     f"Previously verified {target_path.name} is missing or changed")
            replacement = torch.load(target_path, map_location="cpu", weights_only=False)
            _validate_checkpoint(replacement, role, summary, config, contract, torch, compact=True)
            _require(replacement["step"] == record["step"] and replacement["tokens_seen"] == record["tokens_seen"],
                     "Retained checkpoint counters differ from its proof")
            records[role] = record
            del replacement
        gc.collect()

    # No deletion occurs before every replacement has passed verification.
    _require(_sha256(config_path) == config_digest and _sha256(summary_path) == summary_digest,
             "Run metadata changed during compaction; originals retained")
    report = {
        "schema_version": "sparsedrop_400m_retention_v1", "status": "verified",
        "config_sha256": config_digest, "completed_tokens": TOKEN_BUDGET,
        "parameters_total": contract.parameters, "weights_only": True, "resumable": False,
        "original_checkpoint_paths": (previous or {}).get("original_checkpoint_paths", {
            "last_checkpoint": summary.get("last_checkpoint"), "best_checkpoint": summary.get("best_checkpoint")}),
        "verified_utc": (previous or {}).get("verified_utc", datetime.now(timezone.utc).isoformat()),
        "checkpoints": records,
    }
    _write_json(report_path, report)
    updated_summary = deepcopy(summary)
    updated_summary["checkpoint_retention"] = {
        "report": str(report_path), "weights_only": True, "resumable": False,
        "original_checkpoint_paths": report["original_checkpoint_paths"],
    }
    updated_summary["last_checkpoint"] = str(output / "last_model.pt")
    updated_summary["best_checkpoint"] = str(output / "best_model.pt")
    _write_json(summary_path, updated_summary)

    for role in ("best", "last"):
        source_path, target_path = output / f"{role}.pt", output / f"{role}_model.pt"
        # Repeat disk integrity checks immediately before removal. A later
        # interruption can recover using the durable 'verified' report.
        _require(_sha256(config_path) == config_digest, "Configuration changed before source removal")
        _require(_sha256(target_path) == records[role]["replacement_sha256"],
                 f"Replacement changed before source removal: {role}")
        if source_path.exists():
            _require(role in source_identities and _identity(source_path) == source_identities[role],
                     f"Source changed before removal: {role}")
            source_path.unlink()
            _fsync_directory(output)
    report["status"] = "complete"
    report["completed_utc"] = (previous or {}).get("completed_utc", datetime.now(timezone.utc).isoformat())
    report["removed_training_state"] = sorted(EXCLUDED_STATE)
    report["source_checkpoints_removed"] = True
    report["retained_bytes"] = sum(row["replacement_bytes"] for row in records.values())
    report["reclaimed_bytes"] = sum(row["source_identity"]["bytes"] - row["replacement_bytes"]
                                     for row in records.values())
    _write_json(report_path, report)
    return report


def _compact(output_dir: Path, *, contract: _ModelContract) -> dict[str, Any]:
    """Internal contract injection supports small CPU fixtures, never CLI overrides."""
    output = Path(output_dir).resolve(strict=True)
    _require(output.is_dir(), "output_dir must be an existing directory")
    with (output / ".checkpoint_compaction.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another checkpoint compaction is already running") from exc
        return _compact_locked(output, contract)


def compact(output_dir: Path) -> dict[str, Any]:
    return _compact(output_dir, contract=_PRODUCTION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compact(args.output_dir)
    except Exception as exc:
        print(f"Checkpoint compaction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"event": "SPARSEDROP_400M_CHECKPOINT_RETENTION", "status": report["status"],
                      "output_dir": str(args.output_dir.resolve()),
                      "reclaimed_bytes": report["reclaimed_bytes"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
