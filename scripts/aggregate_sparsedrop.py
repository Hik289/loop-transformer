#!/usr/bin/env python3
"""Aggregate SparseDrop runs conservatively; partial runs are never final results.

Example: python scripts/aggregate_sparsedrop.py runs --expected-seeds 42
Writes analysis.json and README_results.md in --output-dir (default: runs).
The script uses only the Python standard library and can run on a laptop.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TREATMENT_KEYS = {"mode", "p", "schedule", "delta"}
MODEL_TREATMENT_KEYS = {"dropout", "dropout_mode", "dropout_location", "dropout_rate", "dropout_schedule", "dropout_delta", "residual_dropout_mode", "residual_dropout_p", "residual_dropout_schedule", "residual_dropout_delta"}
NON_SCIENTIFIC_KEYS = {"config", "output_dir", "resume", "checkpoint_every", "log_every", "device", "max_seconds"}
ARCHIVE_DIRECTORY_NAMES = {"archive", "archives", "archived", "backup", "backups", "recovery"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows, warnings = [], []
    if not path.exists():
        return rows, warnings
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append(f"{path.name}:{index}: incomplete/invalid JSON row omitted")
    return rows, warnings


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def effective_target(config: dict) -> dict:
    """Older runs retain their original budget; capped runs record both targets."""
    if "effective_target_steps" in config or "effective_target_tokens" in config:
        return {"step": config.get("effective_target_steps"), "tokens_seen": config.get("effective_target_tokens")}
    return {"step": config.get("arguments", {}).get("max_steps"), "tokens_seen": config.get("token_budget")}


def comparison_config(config: dict, *, across_seeds: bool = False) -> dict:
    """Compare every scientific setting except the intended dropout treatment."""
    args = config.get("arguments", {})
    excluded = TREATMENT_KEYS | NON_SCIENTIFIC_KEYS
    if across_seeds:
        excluded |= {"seed", "data_seed"}
    result = {
        "arguments": {key: value for key, value in args.items() if key not in excluded
                      and not (key == "stop_after_steps" and value is None)},
        "model": {key: value for key, value in config.get("model", {}).items() if key not in MODEL_TREATMENT_KEYS},
        "optimizer": config.get("optimizer"),
        "token_budget": config.get("token_budget"),
        "effective_target": effective_target(config),
        "data": config.get("data"),
        "data_manifest": config.get("data_manifest", config.get("dataset_manifest")),
        "objective": config.get("objective"),
        "parameter_dtype": config.get("parameter_dtype"),
        "compute_dtype": config.get("compute_dtype"),
        "torch_version": config.get("torch_version"),
        "validation_chunk_indices": config.get("validation_chunk_indices"),
    }
    if not across_seeds:
        result["initial_parameters_sha256"] = config.get("initial_parameters_sha256")
    return result


def condition(config: dict) -> dict:
    args = config.get("arguments", {})
    return {"loops": args.get("loops"), "mode": args.get("mode"), "p": args.get("p"),
            "schedule": args.get("schedule"), "delta": args.get("delta")}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_manifest(value) -> bool:
    return isinstance(value, dict) and bool(value) and value.get("provenance_status") != "unavailable"


def _mean_std(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) >= 2 else None}


def load_run(directory: Path) -> dict:
    config = _read_json(directory / "config.json")
    records, warnings = _jsonl(directory / "eval.jsonl")
    summary_path = directory / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    target = effective_target(config)
    target_valid = all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                       for value in target.values())
    if not target_valid:
        warnings.append("Missing/invalid effective step/token budget; excluded from completed and paired analysis")
    train_records, train_warnings = _jsonl(directory / "train.jsonl")
    warnings.extend(train_warnings)
    committed_events = {"train", "checkpoint", "resume", "start", "interrupted", "complete"}
    progress_records = [summary, *records, *[row for row in train_records if row.get("event") in committed_events]]
    progress = {key: max((row[key] for row in progress_records if _finite_number(row.get(key))), default=None)
                for key in ("step", "tokens_seen")}
    budget_exceeded = summary.get("status") == "stopped_over_budget" or (
        target_valid and any(_finite_number(progress[key]) and progress[key] > target[key] for key in target))
    if budget_exceeded:
        warnings.append(f"Run exceeds its effective budget: actual step={progress['step']}, tokens_seen={progress['tokens_seen']}; "
                        f"target step={target['step']}, tokens_seen={target['tokens_seen']}. Excluded from final and paired analysis")
    complete = summary.get("status") == "complete" and isinstance(summary.get("final_full_validation"), dict)
    if summary.get("status") == "complete" and not complete:
        warnings.append("Missing final full validation; claimed completion is excluded")
    if isinstance(summary.get("final_full_validation"), dict):
        evaluation = {**summary["final_full_validation"], "event": "validation_full_final",
                      "step": summary.get("step"), "tokens_seen": summary.get("tokens_seen")}
    else:
        valid_records = [row for row in records if row.get("event") in {"validation_fixed", "validation_full_final"}]
        evaluation = valid_records[-1] if valid_records else None
    if complete and evaluation is None:
        warnings.append("Run claims completion but has no evaluation; excluded from completed analysis")
        complete = False
    if complete and (not target_valid or any(summary.get(key) != value for key, value in target.items())):
        warnings.append("Completed run does not meet its effective step/token budget; excluded")
        complete = False
    if budget_exceeded:
        complete = False
    if evaluation is not None and not all(_finite_number(evaluation.get(k)) for k in ("loss", "perplexity")):
        warnings.append("Invalid/missing validation quality metric; excluded from analysis")
        evaluation = None
        complete = False
    if not _valid_manifest(config.get("data_manifest", config.get("dataset_manifest"))):
        warnings.append("Dataset/tokenizer manifest missing: file signatures alone do not establish tokenizer provenance")
    return {"directory": str(directory.resolve()), "config": config, "condition": condition(config),
            "seed": config.get("arguments", {}).get("seed"), "complete": complete,
            "status": "stopped_over_budget" if budget_exceeded else "complete" if complete else "incomplete_claimed_completion" if summary.get("status") == "complete" else summary.get("status", "in_progress_or_not_started"),
            "effective_target": target, "training_progress": progress, "budget_exceeded": budget_exceeded,
            "budget_metadata_valid": target_valid,
            "evaluation": evaluation, "summary": summary, "warnings": warnings}


def pairing_errors(baseline: dict, treatment: dict) -> list[str]:
    errors = []
    for role, run in (("baseline", baseline), ("treatment", treatment)):
        if run.get("budget_exceeded") or run.get("status") == "stopped_over_budget":
            errors.append(f"{role} exceeds effective step/token budget")
        if run.get("budget_metadata_valid") is False:
            errors.append(f"{role} has invalid effective budget metadata")
    a, b = baseline["config"], treatment["config"]
    ca, cb = comparison_config(a), comparison_config(b)
    for key in ca:
        if ca[key] != cb[key]:
            errors.append(f"mismatched {key}")
    for key in ("optimizer", "data", "initial_parameters_sha256"):
        if not ca.get(key):
            errors.append(f"missing {key}")
    if not _valid_manifest(ca.get("data_manifest")):
        errors.append("missing dataset/tokenizer manifest")
    ea, eb = baseline.get("evaluation"), treatment.get("evaluation")
    if ea is None or eb is None:
        errors.append("evaluation unavailable")
        return errors
    for key in ("event", "step", "tokens_seen", "tokens", "chunks"):
        if ea.get(key) != eb.get(key):
            errors.append(f"mismatched evaluation {key}")
        elif ea.get(key) is None:
            errors.append(f"missing evaluation {key}")
    pa = (ea.get("diagnostics") or {}).get("protocol")
    pb = (eb.get("diagnostics") or {}).get("protocol")
    if pa != pb:
        errors.append("mismatched diagnostic protocol/token positions")
    return errors


def diagnostic_scalars(evaluation: dict) -> dict[str, float]:
    diagnostics = evaluation.get("diagnostics") or {}
    values = {}
    for key in ("mean_state_cosine", "mean_update_cosine", "mean_linear_cka"):
        value = (diagnostics.get("cross_loop") or {}).get(key)
        if _finite_number(value):
            values[key] = value
    for state in diagnostics.get("loop_states", []):
        for key in ("state_rms", "update_mean_l2", "update_rms"):
            if _finite_number(state.get(key)):
                values[f"loop{state['loop']}/{key}"] = state[key]
    for activation in diagnostics.get("activations", []):
        prefix = f"loop{activation['loop']}/layer{activation['layer']}/{activation['branch']}"
        for key in ("residual_rms", "top_10pct_energy"):
            if _finite_number(activation.get(key)):
                values[f"{prefix}/{key}"] = activation[key]
        for threshold in activation.get("threshold_sensitivity", []):
            for key in ("near_zero_ratio", "active_ratio"):
                if _finite_number(threshold.get(key)):
                    values[f"{prefix}/{key}/threshold={threshold['threshold']}"] = threshold[key]
        for point in activation.get("topk_energy_curve", []):
            if _finite_number(point.get("energy")):
                values[f"{prefix}/energy/fraction={point['fraction']}"] = point["energy"]
    return values


def aggregate(runs: list[dict], expected_seeds: list[int]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        key = _fingerprint({"condition": run["condition"], "paired_config": comparison_config(run["config"], across_seeds=True)})
        groups[key].append(run)
    output_groups = []
    for fingerprint, members in sorted(groups.items()):
        completed = [run for run in members if run["complete"] and run["evaluation"] is not None]
        seeds = [run["seed"] for run in completed]
        duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
        missing = sorted(set(expected_seeds) - set(seeds))
        # Duplicate folders for one seed must not masquerade as independent runs.
        completed_unique = [run for run in completed if run["seed"] not in duplicates and run["seed"] in expected_seeds]
        validation_scopes = {(run["evaluation"].get("event"), run["evaluation"].get("tokens"), run["evaluation"].get("chunks"), run["evaluation"].get("tokens_seen"), run["evaluation"].get("step")) for run in completed_unique}
        protocols = {_fingerprint((run["evaluation"].get("diagnostics") or {}).get("protocol") or {}) for run in completed_unique}
        aggregate_allowed = len(validation_scopes) <= 1 and len(protocols) <= 1
        eligible = completed_unique if aggregate_allowed else []
        scalar_values: dict[str, list[float]] = defaultdict(list)
        for run in eligible:
            for key, value in diagnostic_scalars(run["evaluation"]).items():
                scalar_values[key].append(value)
        entry = {
            "condition": members[0]["condition"], "protocol_fingerprint": fingerprint,
            "effective_target": effective_target(members[0]["config"]),
            "expected_seeds": expected_seeds, "completed_seeds": sorted(set(seeds)),
            "missing_seeds": missing, "duplicate_seeds_excluded": duplicates,
            "contributing_seeds": sorted(run["seed"] for run in eligible),
            "complete_expected_seed_comparison": not missing and not duplicates and aggregate_allowed,
            "aggregation_status": "ok" if aggregate_allowed else "rejected_mixed_evaluation_scopes_or_protocols",
            "validation_loss": _mean_std([run["evaluation"]["loss"] for run in eligible]),
            "perplexity": _mean_std([run["evaluation"]["perplexity"] for run in eligible]),
            "diagnostics_mean_std": {key: _mean_std(value) for key, value in sorted(scalar_values.items())},
            "cost_mean_std": {key: _mean_std([run["summary"][key] for run in eligible if _finite_number(run["summary"].get(key))])
                              for key in ("train_seconds", "mean_train_tokens_per_second", "approx_train_flops", "peak_gpu_memory_bytes")},
            "runs": [{key: run.get(key) for key in ("directory", "seed", "status", "evaluation", "warnings", "training_progress", "effective_target", "budget_exceeded")} for run in members],
        }
        output_groups.append(entry)
    pairs = []
    baselines = [run for run in runs if run["condition"].get("p") == 0 and run["condition"].get("mode") == "none"]
    for treatment in runs:
        if treatment in baselines:
            continue
        candidates = [base for base in baselines if base["seed"] == treatment["seed"] and base["condition"].get("loops") == treatment["condition"].get("loops")]
        valid = [(base, pairing_errors(base, treatment)) for base in candidates]
        matches = [base for base, errors in valid if not errors]
        pair = {"treatment": treatment["directory"], "seed": treatment["seed"], "condition": treatment["condition"],
                "baseline": None, "validation_loss_improvement": None, "status": "unavailable"}
        if len(matches) == 1:
            base = matches[0]
            pair.update(baseline=base["directory"], validation_loss_improvement=base["evaluation"]["loss"] - treatment["evaluation"]["loss"],
                        status="final_paired" if base["complete"] and treatment["complete"] else "interim_paired_not_final")
        elif len(matches) > 1:
            pair["reason"] = "multiple matching baselines; ambiguous duplicate runs"
        else:
            pair["reason"] = [{"baseline": base["directory"], "errors": errors} for base, errors in valid] or "matching baseline seed/H unavailable"
        pairs.append(pair)
    final_pairs = {pair["treatment"]: pair["validation_loss_improvement"] for pair in pairs if pair["status"] == "final_paired"}
    for group in output_groups:
        group["paired_loss_improvement"] = _mean_std([
            final_pairs[run["directory"]] for run in group["runs"]
            if run["directory"] in final_pairs and run["seed"] in expected_seeds
            and run["seed"] not in group["duplicate_seeds_excluded"]
        ] if group["aggregation_status"] == "ok" else [])
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "expected_seeds": expected_seeds,
            "number_of_runs": len(runs), "groups": output_groups, "paired_comparisons": pairs,
            "interpretation": "Only runs completed exactly at their effective step/token budget contribute to final means. Over-budget runs are excluded from final and paired analysis; their actual progress is retained. Std is sample std (ddof=1), unavailable for n<2. Partial paired deltas are exploratory, not conclusions. H1 cross-loop metrics are N/A. No treatment-specific threshold recalibration."}


def render_markdown(report: dict) -> str:
    def tokens_b(value):
        return f"{value / 1e9:.9f}".rstrip("0").rstrip(".") + "B" if _finite_number(value) else "未知"

    lines = ["# SparseDrop-Loop 结果与配对检查", "", f"生成时间（UTC）：{report['generated_utc']}", "",
             "仅恰好达到当前有效步数与 token 预算、且完成全量验证的运行计入最终均值；超过预算的运行不纳入最终均值或配对。原始学习率调度预算保留。种子不足时明确标注；一个种子不报告标准差。训练中途的验证值仅用于监控，不能用于最终效果结论。", "",
             "正文 token 数以 B 表示（1B = 10⁹）；analysis.json 保留原始整数计数。", "",
             "| H | 位置 | p | 调度 | 目标步数 | 目标训练量 | 纳入种子 / n | 缺少种子 | 排除重复种子 | 结果完整 | 最终 validation loss（mean ± sample std） | 最终 PPL |", "|---:|---|---:|---|---:|---:|---|---|---|---|---|---|"]
    def display(stat):
        if stat["mean"] is None:
            return "待完成"
        suffix = f" ± {stat['sample_std']:.5f}" if stat["sample_std"] is not None else "（n=1，std N/A）"
        return f"{stat['mean']:.5f}{suffix}"
    for group in report["groups"]:
        c = group["condition"]
        lines.append(f"| {c['loops']} | {c['mode']} | {c['p']} | {c['schedule']} δ={c['delta']} | {group['effective_target']['step']} | {tokens_b(group['effective_target'].get('tokens_seen'))} | {group['contributing_seeds']} / {group['validation_loss']['n']} | {group['missing_seeds']} | {group['duplicate_seeds_excluded']} | {group['complete_expected_seed_comparison']} | {display(group['validation_loss'])} | {display(group['perplexity'])} |")
    lines.extend(["", "## 运行状态与当前验证", ""])
    for group in report["groups"]:
        for run in group["runs"]:
            ev = run["evaluation"]
            value = (f"step={ev.get('step')}，{ev.get('event')}，loss={ev['loss']:.5f}，PPL={ev['perplexity']:.5f}" if ev else "尚无验证结果")
            progress = run.get("training_progress") or {}
            actual = f"实际已训练 step={progress['step']}，训练量={tokens_b(progress.get('tokens_seen'))}；" if progress.get("step") is not None else ""
            lines.append(f"- `{Path(run['directory']).name}`：{run['status']}；{actual}{value}。")
            for warning in run["warnings"]:
                readable_warning = re.sub(r"tokens_seen=(\d+)", lambda m: "tokens_seen=" + tokens_b(int(m.group(1))), warning)
                lines.append(f"  - 检查提示：{readable_warning}")
    lines.extend(["", "## 与同种子基线的配对", ""])
    for pair in report["paired_comparisons"]:
        delta = pair["validation_loss_improvement"]
        detail = f"Δloss = {delta:.6f}（正值表示 dropout 更优）" if delta is not None else f"未计算：{json.dumps(pair.get('reason'), ensure_ascii=False)}"
        lines.append(f"- `{Path(pair['treatment']).name}`：{pair['status']}；{detail}。")
    lines.extend(["", "## 解释限制", "", "固定验证子集与最终全量验证分开记录；不会混合平均。诊断使用预先固定的有界验证 token 子集（默认最多 512 个），默认统一近零阈值为 0.001、0.01、0.1，实际参数见 analysis.json 的 protocol；每个 p=0 基线的 residual RMS 用于记录尺度参考，不重新拟合处理组阈值。H=1 的跨循环指标为 N/A。相似度下降不等于质量提高，需与 loss/PPL 联合解释。", "", "配对检查包括模型、目标、精度、有效 batch、token 预算、优化器、数据/tokenizer manifest、初始参数哈希、同种子数据顺序、训练进度、验证范围和诊断协议。不匹配的运行不会计算基线差值。近似 FLOPs 不是硬件实测 FLOPs；固定 tokens 的跨 H 结果不等价于匹配 FLOPs 实验。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-seeds", default="42")
    args = parser.parse_args()
    expected = [int(value.strip()) for value in args.expected_seeds.split(",")]
    if not expected or len(expected) != len(set(expected)):
        parser.error("--expected-seeds must contain unique integer seeds")
    runs, read_errors = [], []
    for config_path in sorted(args.runs_dir.rglob("config.json")):
        parts = config_path.relative_to(args.runs_dir).parts[:-1]
        if any(part in ARCHIVE_DIRECTORY_NAMES or part.startswith(("before_100k_cap_", "backup_", "archive_")) for part in parts):
            continue
        try:
            runs.append(load_run(config_path.parent))
        except (OSError, ValueError, TypeError) as exc:
            read_errors.append({"directory": str(config_path.parent), "error": str(exc)})
    report = aggregate(runs, expected)
    report["read_errors"] = read_errors
    output = args.output_dir or args.runs_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "README_results.md").write_text(render_markdown(report))
    print(f"Read {len(runs)} runs; {len(read_errors)} read errors. Wrote {output / 'README_results.md'}")


if __name__ == "__main__":
    main()
