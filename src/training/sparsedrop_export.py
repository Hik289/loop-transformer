"""Validate and export final residual sparsity without loading model tensors.

The source remains ``summary.json.final_full_validation.diagnostics``. These
exports describe post-RMSNorm residual branch outputs, not SwiGLU intermediate
neurons. This module deliberately uses only the Python standard library so it
can also backfill results on machines without PyTorch or a GPU.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any


ACTIVATION_LOCATION = "post_sublayer_RMSNorm_before_dropout_and_residual_addition"
REQUIRED_THRESHOLDS = (0.001, 0.01, 0.1)
BRANCHES = ("attention", "ffn")


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0 or (ratio and value > 1 + 1e-12):
        raise ValueError(f"{label} is outside its valid range")
    return float(value)


def validate_final_diagnostics(
    diagnostics: dict[str, Any], *, loops: int, num_layers: int,
    hidden_size: int | None = None,
) -> list[dict[str, Any]]:
    """Require every configured loop/layer/branch exactly once; return sorted rows.

    The collector's first observed callback set is not sufficient evidence of
    complete coverage. This check derives the full expected set independently
    from the experiment's model dimensions before completion is declared.
    """
    _integer(loops, "loops", 1)
    _integer(num_layers, "num_layers", 1)
    if hidden_size is not None:
        _integer(hidden_size, "hidden_size", 1)
    if not isinstance(diagnostics, dict) or diagnostics.get("status") != "ok":
        raise ValueError("Final diagnostics must exist with status=ok")
    protocol = diagnostics.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Final diagnostics must include a protocol")
    if protocol.get("model_mode") != "eval" or protocol.get("activation_location") != ACTIVATION_LOCATION:
        raise ValueError("Final diagnostics must use eval post-RMSNorm residual outputs")
    sampled = _integer(protocol.get("sampled_tokens"), "protocol.sampled_tokens", 1)
    maximum = _integer(protocol.get("max_tokens"), "protocol.max_tokens", 1)
    if sampled > maximum:
        raise ValueError("protocol.sampled_tokens exceeds protocol.max_tokens")
    thresholds = protocol.get("activation_thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("protocol.activation_thresholds must be a nonempty list")
    for threshold in thresholds:
        _number(threshold, "protocol.activation_thresholds")
    if len(set(thresholds)) != len(thresholds) or not set(REQUIRED_THRESHOLDS).issubset(thresholds):
        raise ValueError("Final diagnostics require unique thresholds including 0.001, 0.01, 0.1")
    fractions = protocol.get("topk_fractions")
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("protocol.topk_fractions must be a nonempty list")
    for fraction in fractions:
        if not 0 < _number(fraction, "protocol.topk_fractions", ratio=True) <= 1:
            raise ValueError("Top-k fractions must lie in (0, 1]")
    if len(set(fractions)) != len(fractions) or 0.1 not in fractions:
        raise ValueError("Final diagnostics require unique top-k fractions including 0.1")
    rows = diagnostics.get("activations")
    if not isinstance(rows, list):
        raise ValueError("Final diagnostics.activations must be a list")
    expected = {(loop, layer, branch) for loop in range(1, loops + 1)
                for layer in range(num_layers) for branch in BRANCHES}
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each activation row must be an object")
        loop = _integer(row.get("loop"), "activation.loop", 1)
        layer = _integer(row.get("layer"), "activation.layer")
        branch = row.get("branch")
        if branch not in BRANCHES:
            raise ValueError(f"Unknown residual branch: {branch!r}")
        key = (loop, layer, branch)
        if key in observed:
            raise ValueError(f"Duplicate activation key: {key}")
        observed.add(key)
        if _integer(row.get("tokens"), f"{key}.tokens", 1) != sampled:
            raise ValueError(f"{key}.tokens differs from protocol.sampled_tokens")
        width = _integer(row.get("hidden_size"), f"{key}.hidden_size", 1)
        if hidden_size is not None and width != hidden_size:
            raise ValueError(f"{key}.hidden_size differs from configured hidden_size")
        _number(row.get("residual_rms"), f"{key}.residual_rms")
        top10 = _number(row.get("top_10pct_energy"), f"{key}.top_10pct_energy", ratio=True)
        sensitivity = row.get("threshold_sensitivity")
        if not isinstance(sensitivity, list):
            raise ValueError(f"{key}.threshold_sensitivity must be a list")
        seen_thresholds = set()
        for point in sensitivity:
            if not isinstance(point, dict):
                raise ValueError(f"{key} threshold entries must be objects")
            threshold = _number(point.get("threshold"), f"{key}.threshold")
            if threshold in seen_thresholds:
                raise ValueError(f"{key} has a duplicate threshold")
            seen_thresholds.add(threshold)
            near_zero = _number(point.get("near_zero_ratio"), f"{key}.near_zero_ratio", ratio=True)
            active = _number(point.get("active_ratio"), f"{key}.active_ratio", ratio=True)
            if not math.isclose(near_zero + active, 1.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{key} near-zero and active ratios must sum to one")
        if seen_thresholds != set(thresholds):
            raise ValueError(f"{key} threshold coverage differs from protocol")
        curve = row.get("topk_energy_curve")
        if not isinstance(curve, list):
            raise ValueError(f"{key}.topk_energy_curve must be a list")
        seen_fractions = set()
        for point in curve:
            if not isinstance(point, dict):
                raise ValueError(f"{key} top-k entries must be objects")
            fraction = _number(point.get("fraction"), f"{key}.fraction", ratio=True)
            if fraction in seen_fractions:
                raise ValueError(f"{key} has a duplicate top-k fraction")
            seen_fractions.add(fraction)
            if _integer(point.get("k"), f"{key}.k", 1) != math.ceil(width * fraction):
                raise ValueError(f"{key}.k differs from ceil(hidden_size * fraction)")
            energy = _number(point.get("energy"), f"{key}.energy", ratio=True)
            if fraction == 0.1 and not math.isclose(energy, top10, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{key} top-10% energy disagrees with its curve")
        if seen_fractions != set(fractions):
            raise ValueError(f"{key} top-k coverage differs from protocol")
    if observed != expected:
        raise ValueError(f"Incomplete final activation coverage: missing={sorted(expected - observed)}, "
                         f"extra={sorted(observed - expected)}")
    return sorted(rows, key=lambda row: (row["loop"], row["layer"], row["branch"]))


def build_final_sparsity(summary: dict[str, Any], config: dict[str, Any], *, run_name: str) -> dict[str, Any]:
    """Build a lossless sparsity subset with readable billion-token metadata."""
    if summary.get("status") not in {"complete", "stopped_over_budget"}:
        raise ValueError("A completed or stopped_over_budget final summary is required")
    arguments, model = config.get("arguments", {}), config.get("model", {})
    if not isinstance(arguments, dict) or not isinstance(model, dict):
        raise ValueError("Configuration arguments/model must be objects")
    loops = arguments.get("loops", model.get("max_recurrent_steps"))
    layers = arguments.get("num_layers", model.get("num_layers"))
    width = arguments.get("hidden_size", model.get("hidden_size"))
    final = summary.get("final_full_validation")
    if not isinstance(final, dict):
        raise ValueError("summary.final_full_validation is required")
    diagnostics = final.get("diagnostics")
    rows = validate_final_diagnostics(diagnostics, loops=loops, num_layers=layers, hidden_size=width)
    step = _integer(summary.get("step"), "summary.step")
    actual_steps = _integer(summary.get("actual_steps", step), "summary.actual_steps")
    tokens_seen = _integer(summary.get("tokens_seen"), "summary.tokens_seen")
    actual_tokens = _integer(summary.get("actual_tokens_seen", tokens_seen), "summary.actual_tokens_seen")
    if step != actual_steps or tokens_seen != actual_tokens:
        raise ValueError("Summary actual step/token counts disagree with step/tokens_seen")
    target_steps = summary.get("effective_target_steps", config.get("effective_target_steps"))
    target_tokens = summary.get("effective_target_tokens", config.get("effective_target_tokens"))
    if target_steps is not None:
        _integer(target_steps, "effective_target_steps", 1)
    if target_tokens is not None:
        _integer(target_tokens, "effective_target_tokens", 1)
    over_budget = (summary["status"] == "stopped_over_budget"
                   or (target_steps is not None and actual_steps > target_steps)
                   or (target_tokens is not None and tokens_seen > target_tokens))
    payload = {"schema_version": "sparsedrop_final_export_v1", "run_name": run_name,
               "source": "summary.json.final_full_validation.diagnostics",
               "checkpoint_scope": "final_model_at_actual_steps", "status": summary["status"],
               "step": step, "actual_steps": actual_steps, "tokens_seen": tokens_seen,
               "actual_tokens_seen": actual_tokens, "training_tokens_B": tokens_seen / 1_000_000_000,
               "completion_reason": summary.get("completion_reason"), "budget_exceeded": over_budget,
               "effective_target_steps": target_steps, "effective_target_tokens": target_tokens,
               "effective_target_tokens_B": None if target_tokens is None else target_tokens / 1_000_000_000,
               "loops": loops, "num_layers": layers, "hidden_size": width,
               "coverage": {"status": "complete", "expected_rows": loops * layers * len(BRANCHES),
                            "observed_rows": len(rows), "branches": list(BRANCHES)},
               "protocol": deepcopy(diagnostics["protocol"]), "activations": deepcopy(rows)}
    for name in ("compute_tokens_seen", "original_token_budget", "overshoot_tokens"):
        if name in summary:
            count = _integer(summary[name], f"summary.{name}")
            payload[name] = count
            payload[name + "_B"] = count / 1_000_000_000
    if "overshoot_steps" in summary:
        payload["overshoot_steps"] = _integer(summary["overshoot_steps"], "summary.overshoot_steps")
    if "token_budget_completed" in summary:
        payload["token_budget_completed"] = summary["token_budget_completed"]
    return payload


def render_final_sparsity(payload: dict[str, Any]) -> str:
    """Render every loop/layer/branch, with B units for training token counts."""
    protocol = payload["protocol"]
    lines = [f"# 最终激活稀疏性：{payload['run_name']}", "",
             f"- 训练状态：`{payload['status']}`；实际步数：{payload['actual_steps']:,}。",
             f"- 已训练 tokens：{payload['training_tokens_B']:.6f} B（1 B = 10⁹ tokens）。",
             f"- 来源：`{payload['source']}`，对应最终模型的实际步数。",
             f"- 完整性：{payload['loops']} loops × {payload['num_layers']} 层 × 2 分支 = "
             f"{payload['coverage']['observed_rows']} 条，覆盖全部配置项且无重复。", ""]
    if payload["effective_target_steps"] is not None:
        lines.append(f"当前目标：{payload['effective_target_steps']:,} 步。")
    if payload["effective_target_tokens_B"] is not None:
        lines.append(f"当前目标训练量：{payload['effective_target_tokens_B']:.6f} B tokens。")
    if payload["budget_exceeded"]:
        lines.extend(["", "**超出当前预算：保留实际步数与数据；不得作为恰好达到当前预算的结果进行配对比较。**"])
    lines.extend(["", f"诊断样本：固定验证顺序的前 {protocol['sampled_tokens']:,} 个有效 positions；"
                  "该数量是激活诊断样本量，不是完整验证集规模。",
                  "测量在 eval 模式下执行，Dropout 关闭。Attention 和 FFN 均取分支输出经 RMSNorm "
                  "之后、Dropout 及残差相加之前的激活；FFN 指标不表示 SwiGLU 中间神经元稀疏性。",
                  "loop 从 1 开始，物理层 layer 从 0 开始。近零比例为 |a| ≤ ε 的比例；"
                  "Top-10% 能量是每个 position 中幅值最大的 ceil(0.1d) 个通道能量占比的平均值。", "",
                  "| Loop | Layer | Branch | Positions | ε=0.001 近零比例 | ε=0.01 近零比例 | ε=0.1 近零比例 | Top-10% 能量 | RMS |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["activations"]:
        sensitivity = {point["threshold"]: point["near_zero_ratio"] for point in row["threshold_sensitivity"]}
        ratios = " | ".join(f"{sensitivity[threshold]:.6%}" for threshold in REQUIRED_THRESHOLDS)
        lines.append(f"| {row['loop']} | {row['layer']} | {row['branch']} | {row['tokens']:,} | "
                     f"{ratios} | {row['top_10pct_energy']:.6%} | {row['residual_rms']:.6f} |")
    lines.extend(["", "完整精度、活跃比例、全部 Top-k 能量曲线及原始计数保存在同目录 "
                  "[sparsity_final.json](sparsity_final.json) 中。", ""])
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def export_final_sparsity(
    run_dir: Path, *, summary: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate first, then atomically replace each derived artifact.

    Source summary/config are never modified. Supplying their in-memory values
    allows the trainer to export immediately after saving its final summary.
    """
    run_dir = Path(run_dir)
    if summary is None:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if config is None:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise ValueError("Summary and configuration must be JSON objects")
    payload = build_final_sparsity(summary, config, run_name=run_dir.name)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_final_sparsity(payload)
    _write_atomic(run_dir / "sparsity_final.json", serialized)
    _write_atomic(run_dir / "README_sparsity_final.md", markdown)
    return payload
