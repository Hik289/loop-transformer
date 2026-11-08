#!/usr/bin/env python3
"""Reproducible single-GPU SparseDrop-Loop training with final-loop CE only.

Use --config experiment.json, or explicit arguments. The independent NumPy data
sampler makes paired experiments see identical tokens even when dropout changes
Torch's random-number consumption. Checkpoints include both random streams.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.sparsedrop_export import export_final_sparsity, validate_final_diagnostics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    preliminary, _ = pre.parse_known_args(argv)
    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--mode", choices=("none", "attention", "ffn", "both"), default="none")
    p.add_argument("--p", type=float, default=0.0)
    p.add_argument("--schedule", choices=("constant", "increasing", "decreasing"), default="constant")
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--token-budget", type=int, default=None, help="Exact number of supervised prediction tokens; overrides max_steps")
    p.add_argument("--stop-after-steps", type=int, default=None, help="Stop at this total optimizer-step count, including resumed steps; preserves the original token budget and learning-rate horizon")
    p.add_argument("--max-seconds", type=float, default=None, help="Save and return 99 after a completed step once this invocation reaches its wall-time limit")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-chunks", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--diagnostic-tokens", type=int, default=512)
    p.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    p.add_argument("--vocab-size", type=int, default=49152)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--intermediate-size", type=int, default=2048)
    if preliminary.config is not None:
        config_values = json.loads(preliminary.config.read_text())
        if not isinstance(config_values, dict):
            p.error("Configuration must be a JSON object")
        unknown = set(config_values) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        p.set_defaults(**config_values)
    args = p.parse_args(argv)
    if args.data_dir is None or args.output_dir is None:
        p.error("--data-dir and --output-dir are required (or provide them in JSON)")
    args.data_dir, args.output_dir = Path(args.data_dir), Path(args.output_dir)
    for key in ("loops", "max_steps", "batch_size", "grad_accum", "seq_len", "eval_every", "eval_chunks", "eval_batch_size", "checkpoint_every", "log_every", "diagnostic_tokens"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.token_budget is not None:
        if args.token_budget <= 0:
            p.error("token_budget must be positive")
        capacity = args.batch_size * args.grad_accum * args.seq_len
        args.max_steps = (args.token_budget + capacity - 1) // capacity
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        p.error("max_seconds must be finite and positive")
    if args.stop_after_steps is not None and (type(args.stop_after_steps) is not int or args.stop_after_steps <= 0):
        p.error("stop_after_steps must be a positive integer")
    if args.warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1:
        p.error("warmup_steps must be nonnegative and min_lr_ratio in [0, 1]")
    if args.lr <= 0 or args.weight_decay < 0 or args.clip_grad <= 0:
        p.error("lr/clip_grad must be positive; weight_decay nonnegative")
    if args.data_seed is None:
        args.data_seed = args.seed + 100_003
    return args


class TokenChunks:
    """Contiguous length-S inputs and exactly one-position-shifted targets."""

    def __init__(self, path: Path, seq_len: int, vocab_size: int):
        self.path, self.seq_len = path, seq_len
        self.tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.tokens.ndim != 1 or not np.issubdtype(self.tokens.dtype, np.integer):
            raise ValueError(f"{path}: expected one-dimensional integer token array")
        self.n_chunks = (len(self.tokens) - 1) // seq_len
        if self.n_chunks < 1:
            raise ValueError(f"{path}: requires at least seq_len + 1 tokens")
        # Scan in bounded chunks; invalid IDs must fail before costly GPU work.
        for offset in range(0, len(self.tokens), 4_000_000):
            part = self.tokens[offset:offset + 4_000_000]
            if part.min() < 0 or part.max() >= vocab_size:
                raise ValueError(f"{path}: token IDs outside [0, {vocab_size})")

    def batch(self, indices: np.ndarray | list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = np.stack([self.tokens[int(i) * self.seq_len:int(i) * self.seq_len + self.seq_len + 1] for i in indices]).astype(np.int64)
        inputs = torch.from_numpy(rows[:, :-1].copy())
        targets = torch.from_numpy(rows[:, 1:].copy())
        return inputs.to(device), targets.to(device)

    def signature(self) -> dict[str, Any]:
        """Identity checks for resume; full dataset hashes live in data manifest."""
        st = self.path.stat()
        return {"path": str(self.path.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "tokens": len(self.tokens), "dtype": str(self.tokens.dtype), "chunks": self.n_chunks, "prediction_tokens": self.n_chunks * self.seq_len, "trailing_prediction_tokens_excluded": len(self.tokens) - 1 - self.n_chunks * self.seq_len}


class ChunkSampler:
    """Shuffled epochs, independent of model/dropout RNG, with exact resume."""

    def __init__(self, n_chunks: int, seed: int):
        self.n_chunks = n_chunks
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(n_chunks)
        self.position, self.epoch = 0, 0

    def take(self, count: int) -> np.ndarray:
        selected = []
        remaining = count
        while remaining:
            if self.position == self.n_chunks:
                self.order = self.rng.permutation(self.n_chunks)
                self.position = 0
                self.epoch += 1
            length = min(remaining, self.n_chunks - self.position)
            selected.append(self.order[self.position:self.position + length])
            self.position += length
            remaining -= length
        return np.concatenate(selected)

    def state_dict(self) -> dict[str, Any]:
        return {"n_chunks": self.n_chunks, "rng": self.rng.bit_generator.state, "order": self.order.copy(), "position": self.position, "epoch": self.epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["n_chunks"] != self.n_chunks:
            raise ValueError("Training chunk count differs from checkpoint")
        self.rng.bit_generator.state = state["rng"]
        self.order = np.array(state["order"], copy=True)
        self.position, self.epoch = state["position"], state["epoch"]


def lr_for_step(step: int, max_steps: int, warmup_steps: int, peak_lr: float, min_ratio: float) -> float:
    """One-based optimizer step; last step reaches the configured minimum."""
    warmup_steps = min(warmup_steps, max_steps)
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return peak_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def approximate_flops_per_token(args: argparse.Namespace) -> int:
    # Forward+backward matrix multiplies, plus forward recomputation when enabled.
    # Full square attention estimate: causal FlashAttention may execute less work.
    d, m, layers, loops = args.hidden_size, args.intermediate_size, args.num_layers, args.loops
    proj_factor = 8 if args.gradient_checkpointing else 6
    attn_factor = 16 if args.gradient_checkpointing else 12
    return int(proj_factor * loops * layers * (4 * d * d + 3 * d * m) + attn_factor * loops * layers * args.seq_len * d + 6 * d * args.vocab_size)


EXECUTION_PROFILE_KEYS = {"batch_size", "grad_accum", "gradient_checkpointing"}


def authorize_execution_transition(output_dir: Path, checkpoint_path: Path,
                                   state: dict[str, Any], run_config: dict[str, Any],
                                   mismatches: dict[str, Any]) -> dict[str, Any]:
    """Accept only an explicit authorization bound to this exact checkpoint.

    A larger microbatch can change dropout RNG consumption even when the
    optimizer batch is unchanged. This is an audited execution transition,
    never an assertion of bitwise-equivalent training.
    """
    refusal = f"Resume would change the experiment: {mismatches}"
    if not mismatches or not set(mismatches) <= EXECUTION_PROFILE_KEYS:
        raise ValueError(refusal)
    path = output_dir / "execution_transition.json"
    if not path.is_file():
        raise ValueError(refusal + "; no checkpoint-bound execution transition authorization")
    authorization = json.loads(path.read_text())
    required = {"schema_version", "output_dir", "checkpoint_sha256", "checkpoint_step",
                "checkpoint_tokens_seen", "from", "to"}
    if not isinstance(authorization, dict) or not required <= set(authorization):
        raise ValueError("Invalid execution transition authorization schema")
    if (type(authorization["schema_version"]) is not int or authorization["schema_version"] != 1
            or authorization["output_dir"] != str(output_dir.resolve())):
        raise ValueError("Execution transition authorization belongs to another output directory or schema")
    old = {key: state["arguments"].get(key) for key in EXECUTION_PROFILE_KEYS}
    new = {key: run_config.get(key) for key in EXECUTION_PROFILE_KEYS}
    for name, profile, expected in (("from", authorization["from"], old),
                                    ("to", authorization["to"], new)):
        if not isinstance(profile, dict) or set(profile) != EXECUTION_PROFILE_KEYS:
            raise ValueError(f"Invalid execution transition {name} profile")
        if (type(profile["batch_size"]) is not int or profile["batch_size"] <= 0
                or type(profile["grad_accum"]) is not int or profile["grad_accum"] <= 0
                or type(profile["gradient_checkpointing"]) is not bool
                or any(profile[key] != expected[key] or type(profile[key]) is not type(expected[key])
                       for key in EXECUTION_PROFILE_KEYS)):
            raise ValueError(f"Execution transition {name} profile does not match checkpoint/configuration")
    if old["batch_size"] * old["grad_accum"] != new["batch_size"] * new["grad_accum"]:
        raise ValueError("Execution transition changes the effective optimizer batch")
    for field, checkpoint_field in (("checkpoint_step", "step"),
                                    ("checkpoint_tokens_seen", "tokens_seen")):
        if (type(authorization[field]) is not int
                or authorization[field] != state.get(checkpoint_field)):
            raise ValueError(f"Execution transition {field} does not match checkpoint")
    expected_hash = authorization["checkpoint_sha256"]
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)):
        raise ValueError("Invalid execution transition checkpoint SHA256")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Execution transition checkpoint SHA256 mismatch")
    return {"time_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": expected_hash,
            "step": state["step"], "tokens_seen": state["tokens_seen"],
            "from": old, "to": new, "reason": authorization.get("reason", "Explicit execution optimization"),
            "authorization": str(path.resolve()), "bitwise_trajectory_preserved": False}


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def emit(path: Path, event: dict[str, Any]) -> None:
    event = {"time_utc": datetime.now(timezone.utc).isoformat(), **event}
    line = json.dumps(event, allow_nan=False, separators=(",", ":"))
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def update_result_readme(output_dir: Path, train_log: Path) -> None:
    """Publish completed production results without changing training success."""
    root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().parent != (root / "results_fineweb20b").resolve():
        return
    try:
        from src.training.sparsedrop_readme import update_readme_results

        report = update_readme_results(
            manifest_path=root / "configs/sparsedrop_long/manifest.json",
            results_root=root / "results_fineweb20b",
            readme_path=root / "README_SparseDrop_Loop_Ouro_Metrics_Formulas_CN.md",
            live_readme_path=root / "README_实验结果实时表_CN.md",
            status_path=root / "analysis_fineweb20b/readme_table_status.json",
        )
        errors = [{"name": row["name"], "error": error}
                  for row in report.get("rows", []) for error in row.get("errors", [])]
        emit(train_log, {"event": "readme_updated", "counts": report.get("counts"),
                         "errors": errors})
        if errors:
            print(f"WARNING: {len(errors)} result errors were excluded from README tables",
                  file=sys.stderr, flush=True)
    except Exception as exc:
        # All final results are already saved. A documentation failure must not
        # invalidate training or block the next afterok stage. Pull retries it.
        print(f"WARNING: README update failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        try:
            emit(train_log, {"event": "readme_update_failed",
                             "error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass


def reconcile_history(output_dir: Path, checkpoint_step: int) -> dict[str, Any]:
    """Archive records beyond the restored checkpoint before appending a new attempt.

    A crash can leave a valid metric row ahead of the most recent atomic save,
    or a partially written JSON row. Preserve that evidence outside active logs.
    Final-evaluation/completion records are regenerated even at the same step.
    """
    archive = output_dir / "recovery" / f"{time.time_ns()}-{os.getpid()}"
    archived = {}
    for name in ("train.jsonl", "eval.jsonl"):
        path = output_dir / name
        if not path.exists():
            continue
        lines = path.read_bytes().splitlines(keepends=True)
        cut = len(lines)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
                invalidated = not line.endswith(b"\n") or row.get("step", 0) > checkpoint_step or row.get("event") in {"validation_full_final", "complete"}
            except (ValueError, AttributeError, TypeError):
                invalidated = True
            if invalidated:
                cut = index
                break
        if cut == len(lines):
            continue
        archive.mkdir(parents=True, exist_ok=True)
        with (archive / name).open("wb") as stream:
            stream.write(b"".join(lines[cut:]))
            stream.flush()
            os.fsync(stream.fileno())
        temp = path.with_name(path.name + f".tmp-{os.getpid()}")
        with temp.open("wb") as stream:
            stream.write(b"".join(lines[:cut]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        archived[name] = {"retained_rows": cut, "archived_rows": len(lines) - cut}
    for name in ("summary.json", "sparsity_final.json", "README_sparsity_final.md"):
        final_artifact = output_dir / name
        if final_artifact.exists():
            archive.mkdir(parents=True, exist_ok=True)
            os.replace(final_artifact, archive / name)
            archived[name] = {"reason": "completion must be revalidated by this attempt"}
    if archived:
        write_json_atomic(archive / "recovery.json", {"checkpoint_step": checkpoint_step, "files": archived})
    return {"archive": str(archive) if archived else None, "files": archived}


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def autocast_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model: LoopLM, data: TokenChunks, indices: np.ndarray, args: argparse.Namespace, device: torch.device, diagnostics: bool = True) -> dict[str, Any]:
    was_training = model.training
    rng_before = capture_rng()
    model.eval()
    loss_sum, token_count = 0.0, 0
    collector = None
    if diagnostics and args.diagnostics:
        from src.training.sparsedrop_metrics import SparseDropMetrics
        collector = SparseDropMetrics(max_tokens=args.diagnostic_tokens)
    try:
        for start in range(0, len(indices), args.eval_batch_size):
            inputs, targets = data.batch(indices[start:start + args.eval_batch_size], device)
            collect = collector is not None and start * args.seq_len < args.diagnostic_tokens
            callbacks = {}
            if collect:
                collector.begin_batch(valid_mask=targets != -100)
                callbacks = {"activation_collector": collector.record_activation, "state_collector": collector.record_state}
            with autocast_context(device):
                output = model(inputs, num_steps=args.loops, final_only=True, **callbacks)
                logits = output.logits[-1]
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
            loss_sum += float(loss)
            token_count += targets.numel()
            if collect:
                collector.end_batch()
        mean_loss = loss_sum / token_count
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite validation loss: {mean_loss}")
        result = {"loss": mean_loss, "perplexity": math.exp(mean_loss) if mean_loss < 700 else None, "tokens": token_count, "chunks": len(indices)}
        if collector is not None:
            result["diagnostics"] = collector.compute()
        return result
    finally:
        model.train(was_training)
        restore_rng(rng_before)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    invocation_started = time.perf_counter()
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    token_budget = args.token_budget if args.token_budget is not None else args.max_steps * tokens_per_step
    # Operational early stopping must not retroactively change the LR schedule
    # or make an existing checkpoint incompatible with its scientific config.
    target_steps = min(args.max_steps, args.stop_after_steps) if args.stop_after_steps is not None else args.max_steps
    stop_policy = {"stop_after_steps": args.stop_after_steps, "effective_target_steps": target_steps, "effective_target_tokens": min(token_budget, target_steps * tokens_per_step), "original_max_steps": args.max_steps, "original_token_budget": token_budget, "lr_schedule_max_steps": args.max_steps}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_log, eval_log = args.output_dir / "train.jsonl", args.output_dir / "eval.jsonl"
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA GPU with BF16 support is required; --device cpu is for tiny tests")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train_data = TokenChunks(args.data_dir / "train.npy", args.seq_len, args.vocab_size)
    val_data = TokenChunks(args.data_dir / "validation.npy", args.seq_len, args.vocab_size)
    signatures = {"train": train_data.signature(), "validation": val_data.signature()}
    # This validation subset is fixed for every run, including different seeds.
    val_indices = np.arange(min(args.eval_chunks, val_data.n_chunks), dtype=np.int64)
    sampler = ChunkSampler(train_data.n_chunks, args.data_seed)
    config = LoopLMConfig(vocab_size=args.vocab_size, hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, intermediate_size=args.intermediate_size, max_seq_len=args.seq_len, max_recurrent_steps=args.loops, dropout=0.0, beta_kl=0.0, residual_dropout_mode=args.mode, residual_dropout_p=args.p, residual_dropout_schedule=args.schedule, residual_dropout_delta=args.delta)
    model = LoopLM(config)
    model.gradient_checkpointing = args.gradient_checkpointing
    # No gate or auxiliary objective: the efficient final_only path never uses it.
    for parameter in model.exit_gate.parameters():
        parameter.requires_grad_(False)
    initial_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        initial_hash.update(name.encode())
        initial_hash.update(parameter.detach().numpy().tobytes())
    model.to(device=device, dtype=torch.float32)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    run_config = serializable_args(args)
    manifest_path = next((args.data_dir / name for name in ("manifest.json", "metadata.json") if (args.data_dir / name).is_file()), None)
    data_manifest = json.loads(manifest_path.read_text()) if manifest_path is not None else {"provenance_status": "unavailable"}
    metadata = {"data_manifest": data_manifest, "arguments": run_config, "model": asdict(config), "objective": "final_loop_next_token_cross_entropy", "parameters_total": sum(p.numel() for p in model.parameters()), "parameters_trainable": sum(p.numel() for p in trainable), "initial_parameters_sha256": initial_hash.hexdigest(), "parameter_dtype": "float32", "compute_dtype": "bfloat16" if device.type == "cuda" else "float32", "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "data": signatures, "validation_subset": "first ordered 128 chunks, or eval_chunks if overridden", "validation_chunk_indices": val_indices.tolist(), "token_budget": token_budget, "token_accounting": "tokens_seen counts supervised predictions; compute_tokens_seen includes masked positions in the final sequence", "flops_per_token_estimate": approximate_flops_per_token(args), "flops_estimate_notes": "Approximate training matrix multiplies including backward and configured block checkpoint recomputation, tied full-vocabulary final head, and full-square attention; multiplied by compute_tokens_seen, including the masked tail of a partial final sequence; excludes normalization, dropout, optimizer, evaluation. Causal attention actual FLOPs may be smaller. Not measured hardware FLOPs.", "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay_policy": "matrix weights only; norm vectors excluded"}}
    metadata.update(stop_policy)
    metadata["flops_estimate_notes"] = (
        "Approximate training matrix multiplies including backward and each execution segment's "
        "configured block checkpoint recomputation, tied full-vocabulary final head, and full-square "
        "attention; accumulated per step using compute tokens, including masked tail positions. "
        "Excludes normalization, dropout, optimizer and evaluation. Causal attention actual FLOPs "
        "may be smaller. Not measured hardware FLOPs. flops_per_token_estimate describes the "
        "current execution profile, not all historical tokens.")
    last_path, best_path = args.output_dir / "last.pt", args.output_dir / "best.pt"
    step, best_loss, trained_seconds = 0, float("inf"), 0.0
    tokens_seen, compute_tokens_seen, peak_gpu_memory_bytes = 0, 0, 0
    approx_train_flops = 0
    execution_history: list[dict[str, Any]] = []
    execution_transition = None
    resumed = False
    available_checkpoints = [path for path in (last_path, best_path) if path.is_file()]
    auto_checkpoint = max(available_checkpoints, key=lambda path: path.stat().st_mtime_ns) if available_checkpoints else last_path
    resume_path = auto_checkpoint if args.resume == "auto" else Path(args.resume) if args.resume != "none" else None
    if resume_path is not None and resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        ignored = {"config", "output_dir", "resume", "log_every", "checkpoint_every", "max_seconds", "stop_after_steps"}
        mismatches = {k: (state["arguments"].get(k), v) for k, v in run_config.items() if k not in ignored and state["arguments"].get(k) != v}
        if mismatches:
            execution_transition = authorize_execution_transition(
                args.output_dir, resume_path, state, run_config, mismatches)
        if state["data_signatures"] != signatures:
            raise ValueError("Dataset identity differs from checkpoint; refusing non-identical resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        restore_rng(state["rng"])
        step, best_loss, trained_seconds = state["step"], state["best_loss"], state["trained_seconds"]
        tokens_seen = state.get("tokens_seen", step * tokens_per_step)
        compute_tokens_seen = state.get("compute_tokens_seen", tokens_seen)
        if "approx_train_flops" in state:
            approx_train_flops = state["approx_train_flops"]
        else:
            # Legacy checkpoints have a single historical execution profile.
            approx_train_flops = compute_tokens_seen * approximate_flops_per_token(
                argparse.Namespace(**state["arguments"]))
        if type(approx_train_flops) is not int or approx_train_flops < 0:
            raise ValueError("Invalid cumulative checkpoint FLOP estimate")
        execution_history = list(state.get("execution_history", []))
        peak_gpu_memory_bytes = state.get("peak_gpu_memory_bytes", 0)
        if tokens_seen != min(step * tokens_per_step, token_budget):
            raise ValueError("Checkpoint token count is inconsistent with its completed optimizer steps")
        if execution_transition is not None:
            execution_transition.update(
                compute_tokens_seen=compute_tokens_seen, approx_train_flops=approx_train_flops,
                previous_flops_per_token_estimate=approximate_flops_per_token(
                    argparse.Namespace(**state["arguments"])),
                next_flops_per_token_estimate=metadata["flops_per_token_estimate"])
            execution_history.append(execution_transition)
        recovery = reconcile_history(args.output_dir, step)
        resumed = True
    elif args.resume not in ("auto", "none"):
        raise FileNotFoundError(resume_path)
    elif last_path.exists() or (train_log.exists() and train_log.stat().st_size):
        raise RuntimeError("Output directory has earlier run data without a usable resume; choose a fresh output directory")
    metadata["restored_step"] = step
    metadata["restored_tokens_seen"] = tokens_seen
    metadata["restored_approx_train_flops"] = approx_train_flops
    metadata["execution_history"] = execution_history
    metadata["overshoot_steps"] = max(0, step - target_steps)
    metadata["overshoot_tokens"] = max(0, tokens_seen - stop_policy["effective_target_tokens"])
    write_json_atomic(args.output_dir / "config.json", metadata)
    stop_signal = {"value": None}
    def request_stop(signum, frame):
        stop_signal["value"] = signal.Signals(signum).name
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    def update_peak_memory() -> int:
        nonlocal peak_gpu_memory_bytes
        if device.type == "cuda":
            peak_gpu_memory_bytes = max(peak_gpu_memory_bytes, torch.cuda.max_memory_allocated())
        return peak_gpu_memory_bytes
    def checkpoint_payload() -> dict[str, Any]:
        return {"format_version": 2, "step": step, "best_loss": best_loss, "trained_seconds": trained_seconds, "tokens_seen": tokens_seen, "compute_tokens_seen": compute_tokens_seen, "approx_train_flops": approx_train_flops, "execution_history": execution_history, "peak_gpu_memory_bytes": update_peak_memory(), "arguments": run_config, "model_config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(), "rng": capture_rng(), "data_signatures": signatures, "initial_parameters_sha256": metadata["initial_parameters_sha256"]}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # A hard failure on the first update can now restart from the same directory.
    if not resumed:
        save_checkpoint(last_path, checkpoint_payload())
    else:
        emit(train_log, {"event": "resume", "step": step, "checkpoint": str(resume_path), "tokens_seen": tokens_seen, "recovery": recovery, "max_seconds": args.max_seconds})
    if execution_transition is not None:
        emit(train_log, {"event": "execution_transition", **execution_transition})
    emit(train_log, {"event": "start", "step": step, "config": str(args.output_dir / "config.json"), "token_budget": metadata["token_budget"], "parameters": metadata["parameters_total"], "initial_parameters_sha256": metadata["initial_parameters_sha256"], "device": str(device), "gpu": metadata["gpu"], "data_seed": args.data_seed, "max_seconds": args.max_seconds, **stop_policy})
    model.train()
    rolling_loss, rolling_grad, rolling_count, rolling_seconds = 0.0, 0.0, 0, 0.0
    rolling_tokens = 0
    last_fixed_validation_step = None
    while step < target_steps:
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        step += 1
        lr = lr_for_step(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = min(tokens_per_step, token_budget - tokens_seen)
        remaining_tokens, step_compute_tokens = step_tokens, 0
        while remaining_tokens:
            micro_tokens = min(args.batch_size * args.seq_len, remaining_tokens)
            micro_chunks = (micro_tokens + args.seq_len - 1) // args.seq_len
            inputs, targets = train_data.batch(sampler.take(micro_chunks), device)
            # Only the last sequence can be partial. Causality prevents its
            # unsupervised tail from changing earlier supervised predictions.
            targets.reshape(-1)[micro_tokens:] = -100
            step_compute_tokens += targets.numel()
            with autocast_context(device):
                output = model(inputs, num_steps=args.loops, final_only=True)
                logits = output.logits[-1]
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if not torch.isfinite(loss):
                emit(train_log, {"event": "failure", "step": step, "reason": "nonfinite_training_loss"})
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            weighted_loss = loss / args.grad_accum if step_tokens == tokens_per_step else loss * (micro_tokens / step_tokens)
            step_loss += float(loss.detach()) * (micro_tokens / step_tokens)
            weighted_loss.backward()
            remaining_tokens -= micro_tokens
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad, error_if_nonfinite=True))
        optimizer.step()
        tokens_seen += step_tokens
        compute_tokens_seen += step_compute_tokens
        approx_train_flops += step_compute_tokens * metadata["flops_per_token_estimate"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        trained_seconds += elapsed
        rolling_loss += step_loss * step_tokens
        rolling_grad += grad_norm
        rolling_count += 1
        rolling_seconds += elapsed
        rolling_tokens += step_tokens
        if step == 1 or step % args.log_every == 0 or step == target_steps:
            emit(train_log, {"event": "train", "step": step, "loss": rolling_loss / rolling_tokens, "grad_norm_preclip": rolling_grad / rolling_count, "grad_norm_preclip_last": grad_norm, "clip_grad": args.clip_grad, "lr": lr, "tokens_seen": tokens_seen, "compute_tokens_seen": compute_tokens_seen, "step_prediction_tokens": step_tokens, "tokens_per_second": rolling_tokens / rolling_seconds, "train_seconds": trained_seconds, "step_seconds": elapsed, "peak_gpu_memory_bytes": update_peak_memory(), "approx_train_flops": approx_train_flops, "sampler_epoch": sampler.epoch})
            rolling_loss, rolling_grad, rolling_count, rolling_seconds = 0.0, 0.0, 0, 0.0
            rolling_tokens = 0
        if args.max_seconds is not None and time.perf_counter() - invocation_started >= args.max_seconds:
            stop_signal["value"] = stop_signal["value"] or "max_seconds"
        if step % args.eval_every == 0 or step == target_steps:
            result = evaluate(model, val_data, val_indices, args, device)
            last_fixed_validation_step = step
            emit(eval_log, {"event": "validation_fixed", "step": step, "tokens_seen": tokens_seen, **result})
            if result["loss"] < best_loss:
                best_loss = result["loss"]
                save_checkpoint(best_path, checkpoint_payload())
        if step % args.checkpoint_every == 0 or step == target_steps or stop_signal["value"] is not None:
            save_checkpoint(last_path, checkpoint_payload())
            emit(train_log, {"event": "checkpoint", "step": step, "path": str(last_path), "signal": stop_signal["value"]})
        if stop_signal["value"] is not None and step < target_steps:
            emit(train_log, {"event": "interrupted", "step": step, "signal": stop_signal["value"], "resume": str(last_path)})
            return 99
    # A checkpoint already at or beyond a newly lowered cap gets no optimizer
    # updates. Validate and save its actual step/tokens without rewriting history.
    if last_fixed_validation_step != step:
        result = evaluate(model, val_data, val_indices, args, device)
        emit(eval_log, {"event": "validation_fixed", "step": step, "tokens_seen": tokens_seen, **result})
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            save_checkpoint(best_path, checkpoint_payload())
        save_checkpoint(last_path, checkpoint_payload())
        emit(train_log, {"event": "checkpoint", "step": step, "path": str(last_path), "signal": stop_signal["value"], "reason": "target_already_reached"})
    # Full validation on the final model. The fixed subset selects best.pt only.
    full = evaluate(model, val_data, np.arange(val_data.n_chunks), args, device)
    full["evaluation_scope"] = "all_complete_validation_chunks"
    full["trailing_prediction_tokens_excluded"] = len(val_data.tokens) - 1 - val_data.n_chunks * args.seq_len
    if args.diagnostics:
        validate_final_diagnostics(full.get("diagnostics"), loops=args.loops,
                                   num_layers=args.num_layers, hidden_size=args.hidden_size)
    emit(eval_log, {"event": "validation_full_final", "step": step, "tokens_seen": tokens_seen, **full})
    summary = {"status": "complete", "step": step, "tokens_seen": tokens_seen, "compute_tokens_seen": compute_tokens_seen, "best_fixed_validation_loss": best_loss, "final_full_validation": full, "train_seconds": trained_seconds, "mean_train_tokens_per_second": tokens_seen / trained_seconds, "approx_train_flops": approx_train_flops, "execution_history": execution_history, "last_checkpoint": str(last_path), "best_checkpoint": str(best_path), "peak_gpu_memory_bytes": update_peak_memory()}
    completion = {**stop_policy, "status": "stopped_over_budget" if step > target_steps else "complete", "completion_reason": "stop_after_steps" if target_steps < args.max_steps else "original_budget", "token_budget_completed": tokens_seen >= token_budget, "actual_steps": step, "actual_tokens_seen": tokens_seen, "overshoot_steps": max(0, step - target_steps), "overshoot_tokens": max(0, tokens_seen - stop_policy["effective_target_tokens"])}
    summary.update(completion)
    metadata.update(completion)
    write_json_atomic(args.output_dir / "config.json", metadata)
    write_json_atomic(args.output_dir / "summary.json", summary)
    if args.diagnostics:
        export_final_sparsity(args.output_dir, summary=summary, config=metadata)
    emit(train_log, {"event": "complete", "step": step, "summary": str(args.output_dir / "summary.json"), "full_validation_loss": full["loss"], "full_validation_perplexity": full["perplexity"], **completion})
    update_result_readme(args.output_dir, train_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
