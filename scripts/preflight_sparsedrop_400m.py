#!/usr/bin/env python3
"""Exercise the selected 400M training path inside one allocated CUDA GPU.

This is a synthetic smoke check, never a scientific training result. Every
invocation executes the check; existing reports are not treated as a cache.
Only a newly created TemporaryDirectory is removed on exit. Production data,
results and checkpoints are never passed to the trainer or deleted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS_TOTAL = 404_279_297
PARAMETERS_TRAINABLE = 404_277_248
TOKENS_PER_STEP = 8 * 4 * 512
SCIENTIFIC_CONFIG = {
    "vocab_size": 49152,
    "hidden_size": 2048,
    "num_layers": 6,
    "num_heads": 32,
    "intermediate_size": 5504,
    "loops": 12,
    "mode": "both",
    "p": 0.1,
    "schedule": "constant",
    "delta": 0.0,
    "seed": 42,
    "data_seed": 100045,
    "batch_size": 8,
    "grad_accum": 4,
    "seq_len": 512,
    "gradient_checkpointing": True,
    "token_budget": 20_000_000_000,
    "lr": 3e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "clip_grad": 1.0,
    "diagnostics": True,
    "diagnostic_tokens": 512,
    "eval_chunks": 128,
    "eval_batch_size": 2,
    "device": "cuda",
}


def selected_configuration(config_path: Path | None = None, *, root: Path | None = None,
                           environ: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve queued jobs without requiring changes to their stored Slurm script.

    Registered configurations use the exact profile approved for their index.
    Standalone use without an index or --config retains the original H12 smoke
    check. Production paths are provenance only: synthetic data/checkpoints are
    still created exclusively in the disposable temporary directory.
    """
    root = ROOT if root is None else Path(root)
    environ = os.environ if environ is None else environ
    index_text = environ.get('SLURM_ARRAY_TASK_ID')
    if config_path is None and index_text is None:
        return dict(SCIENTIFIC_CONFIG), {'selection': 'standalone_h12_default'}
    rows = read_json(root / 'configs/sparsedrop_400m20b/manifest.json')
    require(isinstance(rows, list) and len(rows) == 18
            and [row['index'] for row in rows] == list(range(18)), 'Invalid 18-run manifest')
    if config_path is None:
        index = int(index_text)
        require(0 <= index < 18, 'Slurm array index is outside the experiment manifest')
        row = rows[index]
        config_path = (root / row['config']).resolve()
    else:
        config_path = Path(config_path).resolve()
        matches = [row for row in rows if (root / row['config']).resolve() == config_path]
        require(len(matches) == 1, '--config must identify one registered 400M experiment')
        row = matches[0]
        index = row['index']
        if index_text is not None:
            require(index == int(index_text), '--config disagrees with the Slurm array index')
    # This script is also directly invoked as `python scripts/...`, where the
    # repository root is not automatically on sys.path.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.validate_sparsedrop_400m_launch import EXECUTION_KEYS, execution_profile

    actual = read_json(config_path)
    profile = execution_profile(index, root)
    require(all(actual.get(key) == value and type(actual.get(key)) is type(value)
                for key, value in profile.items()),
            'Selected configuration differs from its approved execution profile')
    require(actual.get('loops') == row['loops'] and actual.get('mode') == row['mode'],
            'Selected loops/mode differ from the experiment manifest')
    require(row['loops'] in (2, 4, 6, 8, 10, 12)
            and row['mode'] in ('ffn', 'attention', 'both'), 'Unsupported loops/mode')
    variable = set(EXECUTION_KEYS) | {'loops', 'mode'}
    require(all(actual.get(key) == value for key, value in SCIENTIFIC_CONFIG.items()
                if key not in variable), 'Selected configuration changes the scientific contract')
    scientific = {**SCIENTIFIC_CONFIG, **profile, 'loops': row['loops'], 'mode': row['mode']}
    require(scientific['batch_size'] * scientific['grad_accum'] * scientific['seq_len']
            == TOKENS_PER_STEP, 'Selected profile changes the effective optimizer batch')
    return scientific, {
        'selection': 'registered_config', 'index': index, 'config': str(config_path),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'execution_profile': profile,
        'execution_policy_sha256': hashlib.sha256(
            (root / 'cluster/400m_execution_profiles.json').read_bytes()).hexdigest(),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """A failed rerun cannot leave an older passed report at the target path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_hashes() -> dict[str, str]:
    sources = (
        "scripts/preflight_sparsedrop_400m.py",
        "scripts/validate_sparsedrop_400m_launch.py",
        "scripts/train_sparsedrop.py",
        "src/model/config.py",
        "src/model/looplm.py",
        "src/model/transformer.py",
        "src/model/rope.py",
        "src/training/sparsedrop_metrics.py",
        "src/training/sparsedrop_export.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_checkpoint(path: Path, expected_step: int, torch: Any,
                        scientific_config: dict[str, Any]) -> dict[str, Any]:
    """Load the actual saved file, including Adam state, without allocating GPU memory."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint["step"] == expected_step, "Checkpoint step is incorrect")
    require(checkpoint["tokens_seen"] == expected_step * TOKENS_PER_STEP,
            "Checkpoint token count is incorrect")
    require(checkpoint["compute_tokens_seen"] == expected_step * TOKENS_PER_STEP,
            "Checkpoint compute token count is incorrect")
    for key, expected in scientific_config.items():
        require(checkpoint["arguments"][key] == expected,
                f"Checkpoint scientific configuration differs: {key}")
    tensors_checked = 0
    for name, tensor in checkpoint["model"].items():
        require(bool(torch.isfinite(tensor).all()), f"Nonfinite checkpoint model tensor: {name}")
        tensors_checked += 1
    states = checkpoint["optimizer"]["state"]
    require(bool(states), "Checkpoint contains no Adam optimizer states")
    for parameter_id, state in states.items():
        require(int(state["step"].item()) == expected_step,
                f"Adam state was not advanced to step {expected_step}: {parameter_id}")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state[name]
            require(bool(torch.isfinite(tensor).all()),
                    f"Nonfinite Adam tensor: {parameter_id}/{name}")
            tensors_checked += 1
    result = {
        "loaded": True,
        "step": expected_step,
        "tokens_seen": checkpoint["tokens_seen"],
        "bytes": path.stat().st_size,
        "finite_model_and_adam_tensors_checked": tensors_checked,
        "adam_parameter_states": len(states),
        "initial_parameters_sha256": checkpoint["initial_parameters_sha256"],
    }
    del checkpoint
    gc.collect()
    return result


def validate_summary(output: Path, expected_step: int, validate_diagnostics: Any,
                     scientific_config: dict[str, Any]) -> dict[str, Any]:
    summary, config = read_json(output / "summary.json"), read_json(output / "config.json")
    require(summary["status"] == "complete", "Synthetic stop cap did not complete")
    require(summary["step"] == expected_step, "Summary step is incorrect")
    require(summary["tokens_seen"] == expected_step * TOKENS_PER_STEP,
            "Summary token progress is incorrect")
    require(summary["compute_tokens_seen"] == expected_step * TOKENS_PER_STEP,
            "Summary compute token progress is incorrect")
    require(summary["completion_reason"] == "stop_after_steps"
            and not summary["token_budget_completed"],
            "Synthetic check must not claim to have completed 20B tokens")
    require(summary["original_token_budget"] == 20_000_000_000
            and summary["lr_schedule_max_steps"] == 1_220_704,
            "The smoke check changed the production learning-rate horizon")
    require(config["parameters_total"] == PARAMETERS_TOTAL
            and config["parameters_trainable"] == PARAMETERS_TRAINABLE,
            "The instantiated model is not the planned 404M dense model")
    for key, expected in scientific_config.items():
        require(config["arguments"][key] == expected,
                f"Actual training scientific configuration differs: {key}")
    validation = summary["final_full_validation"]
    require(math.isfinite(validation["loss"]) and math.isfinite(validation["perplexity"]),
            "Validation metrics are nonfinite")
    diagnostics = validation["diagnostics"]
    loops = scientific_config['loops']
    expected_rows = loops * scientific_config['num_layers'] * 2
    rows = validate_diagnostics(diagnostics, loops=loops, num_layers=6, hidden_size=2048)
    require(len(rows) == expected_rows and diagnostics["protocol"]["sampled_tokens"] == 512,
            f"Diagnostics must cover {expected_rows} branches on 512 fixed tokens")
    cross = diagnostics["cross_loop"]
    for name in ("state_cosine", "update_cosine", "linear_cka"):
        matrix = cross[name]
        require(len(matrix) == loops and all(len(row) == loops for row in matrix),
                f"Incomplete loop diagnostic matrix: {name}")
        require(all(math.isfinite(value) for row in matrix for value in row),
                f"Nonfinite loop diagnostic matrix: {name}")
    export = read_json(output / "sparsity_final.json")
    require(export["coverage"]["observed_rows"] == expected_rows and export["step"] == expected_step,
            "Final sparsity export is missing or belongs to another checkpoint")
    return {
        "step": expected_step,
        "tokens_seen": summary["tokens_seen"],
        "parameters_total": config["parameters_total"],
        "parameters_trainable": config["parameters_trainable"],
        "restored_step": config["restored_step"],
        "initial_parameters_sha256": config["initial_parameters_sha256"],
        "finite_validation_loss": validation["loss"],
        "finite_validation_perplexity": validation["perplexity"],
        "diagnostic_rows": len(rows),
        "diagnostic_tokens": diagnostics["protocol"]["sampled_tokens"],
        "sample_position_sha256": diagnostics["protocol"]["sample_position_sha256"],
        "loop_matrix_dimensions": [loops, loops],
        "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
        "trained_seconds": summary["train_seconds"],
    }


def run(report: dict[str, Any], temporary_root: Path, scientific_config: dict[str, Any]) -> None:
    # Keep CLI help usable without importing the GPU environment.
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from scripts.train_sparsedrop import lr_for_step, main as train
    from src.training.sparsedrop_export import validate_final_diagnostics

    require(torch.cuda.is_available(), "An allocated CUDA GPU is required")
    require(torch.cuda.device_count() == 1, "Expose exactly one allocated GPU to this check")
    require(torch.cuda.is_bf16_supported(), "The allocated GPU must support BF16")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    report["gpu"] = {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    require("H100" in properties.name or "H200" in properties.name,
            f"Expected one H100 or H200, received {properties.name}")
    report["slurm"] = {name: os.environ.get(name) for name in (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")}

    with tempfile.TemporaryDirectory(prefix="sparsedrop_400m_preflight_", dir=temporary_root) as temporary:
        smoke = Path(temporary)
        data, output = smoke / "data", smoke / "result"
        data.mkdir()
        rng = np.random.default_rng(900042)
        np.save(data / "train.npy", rng.integers(0, 49152, 4 * TOKENS_PER_STEP + 1, dtype=np.uint16))
        np.save(data / "validation.npy", rng.integers(0, 49152, 2 * 512 + 1, dtype=np.uint16))
        (data / "metadata.json").write_text(json.dumps({
            "dataset": "synthetic_400m_cuda_preflight_only",
            "tokenizer": "random_integer_ids_v49152",
            "scientific_result": False,
        }) + "\n")
        config = {**scientific_config, "data_dir": str(data), "output_dir": str(output),
                  "checkpoint_every": 3, "eval_every": 3, "log_every": 1}
        config_path = smoke / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")

        status = train(["--config", str(config_path), "--stop-after-steps", "3", "--resume", "none"])
        require(status == 0, f"First training phase returned {status}")
        first = validate_summary(output, 3, validate_final_diagnostics, scientific_config)
        require(first["restored_step"] == 0, "First phase unexpectedly resumed earlier data")
        first["checkpoint"] = validate_checkpoint(output / "last.pt", 3, torch, scientific_config)
        report["first_phase"] = first
        gc.collect()
        torch.cuda.empty_cache()

        status = train(["--config", str(config_path), "--stop-after-steps", "4",
                        "--resume", str(output / "last.pt")])
        require(status == 0, f"Resumed training phase returned {status}")
        resumed = validate_summary(output, 4, validate_final_diagnostics, scientific_config)
        require(resumed["restored_step"] == 3, "Second phase did not resume the step-3 checkpoint")
        require(resumed["initial_parameters_sha256"] == first["initial_parameters_sha256"],
                "Initial model identity changed across checkpoint recovery")
        resumed["checkpoint"] = validate_checkpoint(output / "last.pt", 4, torch, scientific_config)
        report["resumed_phase"] = resumed

        train_rows = [row for row in jsonl(output / "train.jsonl") if row["event"] == "train"]
        require([row["step"] for row in train_rows] == [1, 2, 3, 4],
                "Expected exactly four optimizer steps, with no duplicate or skipped step")
        for row in train_rows:
            require(row["tokens_seen"] == row["step"] * TOKENS_PER_STEP,
                    "Training record has incorrect token progress")
            require(row["step_prediction_tokens"] == TOKENS_PER_STEP,
                    "Training record used the wrong effective batch")
            for key in ("loss", "grad_norm_preclip", "grad_norm_preclip_last", "step_seconds", "lr"):
                require(math.isfinite(row[key]), f"Nonfinite train metric: {key}")
            expected_lr = lr_for_step(row["step"], 1_220_704, 2000, 3e-4, 0.1)
            require(math.isclose(row["lr"], expected_lr, rel_tol=1e-12),
                    "Learning rate differs from the production schedule")
        peak = max(first["peak_gpu_memory_bytes"], resumed["peak_gpu_memory_bytes"])
        require(0 < peak < properties.total_memory, "Peak CUDA allocation is invalid")
        report.update({
            "train_steps": [{key: row[key] for key in (
                "step", "tokens_seen", "step_prediction_tokens", "loss", "grad_norm_preclip_last",
                "lr", "step_seconds", "tokens_per_second", "peak_gpu_memory_bytes")}
                for row in train_rows],
            "peak_gpu_memory_bytes": peak,
            "peak_gpu_memory_GiB": peak / 2**30,
            "peak_allocated_fraction_of_device_memory": peak / properties.total_memory,
            "checkpoint_save_load_and_resume_passed": True,
            "per_loop_layer_branch_diagnostics_passed": True,
            "four_optimizer_steps_passed": True,
            "synthetic_training_tokens": 4 * TOKENS_PER_STEP,
        })
    report["temporary_synthetic_data_and_checkpoints_removed"] = True
    gc.collect()
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="Atomic JSON proof; this invocation always reruns the preflight")
    parser.add_argument("--config", type=Path,
                        help="Registered production config; defaults to this Slurm array element")
    parser.add_argument("--tmp-root", type=Path,
                        default=Path(os.environ.get("TMPDIR", str(ROOT / "tmp"))),
                        help="Parent for a newly created disposable smoke-test directory")
    args = parser.parse_args(argv)
    args.report = args.report.expanduser().resolve()
    args.tmp_root = args.tmp_root.expanduser().resolve()
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "sparsedrop_400m_preflight_v2",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_smoke_only": True,
        "scientific_result": False,
        "reused_cached_proof": False,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8,
                      "fused": True, "weight_decay_policy": "matrix weights only"},
        "parameter_dtype": "float32",
        "compute_dtype": "bfloat16",
        "expected_parameters_total": PARAMETERS_TOTAL,
        "expected_parameters_trainable": PARAMETERS_TRAINABLE,
        "tokens_per_optimizer_step": TOKENS_PER_STEP,
        "production_lr_horizon_steps": 1_220_704,
        "source_sha256": source_hashes(),
    }
    write_report(args.report, report)
    try:
        scientific_config, selection = selected_configuration(args.config)
        report['scientific_configuration'] = scientific_config
        report['configuration_selection'] = selection
        write_report(args.report, report)
        run(report, args.tmp_root, scientific_config)
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.perf_counter() - started
    write_report(args.report, report)
    print(json.dumps({"event": "SPARSEDROP_400M_PREFLIGHT", "status": report["status"],
                      "report": str(args.report),
                      "peak_gpu_memory_GiB": report.get("peak_gpu_memory_GiB")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
