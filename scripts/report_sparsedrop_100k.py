#!/usr/bin/env python3
"""Report matched 100k validation records without substituting later checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aggregate_sparsedrop import comparison_config, load_run, pairing_errors


TARGET = 100_000
TOKENS = 1_638_400_000
LABELS = {"none": "Baseline", "attention": "Attention p=0.1", "ffn": "FFN p=0.1", "both": "Both p=0.1"}
ORDER = {"none": 0, "attention": 1, "ffn": 2, "both": 3}


def checked_activations(run, evaluation):
    args = run["config"]["arguments"]
    diag = evaluation["diagnostics"]
    acts = diag["activations"]
    expected = {(h, layer, branch) for h in range(1, args["loops"] + 1)
                for layer in range(args["num_layers"]) for branch in ("attention", "ffn")}
    keys = [(a["loop"], a["layer"], a["branch"]) for a in acts]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert diag["status"] == "ok" and diag["protocol"]["model_mode"] == "eval"
    for a in acts:
        assert a["tokens"] == diag["protocol"]["sampled_tokens"]
        assert a["hidden_size"] == args["hidden_size"]
        for t in a["threshold_sensitivity"]:
            assert math.isfinite(t["near_zero_ratio"]) and 0 <= t["near_zero_ratio"] <= 1
    return acts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    runs = [load_run(d) for d in sorted(args.results.iterdir()) if (d / "config.json").is_file()]
    runs = [r for r in runs if r["condition"]["schedule"] == "constant" and r["condition"]["p"] in (0, 0.1)]
    runs.sort(key=lambda r: (r["condition"]["loops"], ORDER[r["condition"]["mode"]]))
    fixed = {}
    for r in runs:
        if not r["summary"]:
            continue
        records = [json.loads(line) for line in (Path(r["directory"]) / "eval.jsonl").read_text().splitlines() if line.strip()]
        matches = [e for e in records if e.get("event") == "validation_fixed" and e.get("step") == TARGET and e.get("tokens_seen") == TOKENS]
        if matches:
            assert all(e["loss"] == matches[-1]["loss"] for e in matches), "Conflicting duplicate 100k evaluations"
            checked_activations(r, matches[-1])
            fixed[r["directory"]] = matches[-1]
    baselines = {r["condition"]["loops"]: r for r in runs if r["condition"]["mode"] == "none"}
    output = {"generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
              "target_steps": TARGET, "training_tokens_B": TOKENS / 1e9,
              "fixed_validation": [], "full_validation": [], "pending_or_excluded": [], "ffn_loop_means_fixed_100k": []}
    for r in runs:
        baseline = baselines.get(r["condition"]["loops"])
        if baseline is None:
            continue
        assert comparison_config(r["config"]) == comparison_config(baseline["config"]), r["directory"]
        for scope in ("fixed", "full"):
            if scope == "fixed":
                e, b = fixed.get(r["directory"]), fixed.get(baseline["directory"])
            else:
                e = r["evaluation"] if r["complete"] else None
                b = baseline["evaluation"] if baseline["complete"] else None
                if e and b:
                    assert not pairing_errors(baseline, r), pairing_errors(baseline, r)
            if e is None or b is None:
                continue
            for key in ("event", "step", "tokens_seen", "tokens", "chunks"):
                assert e[key] == b[key], (r["directory"], scope, key)
            assert e["step"] == TARGET and e["tokens_seen"] == TOKENS
            assert e["diagnostics"]["protocol"] == b["diagnostics"]["protocol"]
            acts = checked_activations(r, e)
            row = {"H": r["condition"]["loops"], "mode": r["condition"]["mode"],
                   "step": TARGET, "training_tokens_B": TOKENS / 1e9,
                   "validation_tokens_B": e["tokens"] / 1e9,
                   "loss": e["loss"], "PPL": e["perplexity"],
                   "delta_loss": e["loss"] - b["loss"],
                   "delta_PPL": e["perplexity"] - b["perplexity"],
                   "PPL_change_percent": 100 * (e["perplexity"] / b["perplexity"] - 1),
                   "source": str(Path(r["directory"]) / ("eval.jsonl" if scope == "fixed" else "summary.json")),
                   "historical_100k_record_of_overbudget_run": r["budget_exceeded"]}
            output[scope + "_validation"].append(row)
            if scope == "fixed":
                name = Path(r["directory"]).name
                detail = {"source": row["source"], "event": e["event"], "step": TARGET,
                          "training_tokens_B": TOKENS / 1e9, "validation_tokens_B": e["tokens"] / 1e9,
                          "actual_final_steps": r["summary"].get("actual_steps", r["summary"].get("step")),
                          "historical_record": r["budget_exceeded"], "diagnostics": e["diagnostics"]}
                (dest / f"{name}_sparsity_fixed_100k.json").write_text(json.dumps(detail, indent=2) + "\n")
                for loop in range(1, row["H"] + 1):
                    subset = [a for a in acts if a["loop"] == loop and a["branch"] == "ffn"]
                    near = [next(t["near_zero_ratio"] for t in a["threshold_sensitivity"] if t["threshold"] == .01) for a in subset]
                    output["ffn_loop_means_fixed_100k"].append({"H": row["H"], "mode": row["mode"], "loop": loop,
                        "mean_near_zero_eps0.01": sum(near) / len(near),
                        "mean_top10_energy": sum(a["top_10pct_energy"] for a in subset) / len(subset)})
        if not r["complete"]:
            output["pending_or_excluded"].append({"directory": r["directory"], "status": r["status"],
                "actual_steps": r["training_progress"]["step"],
                "actual_training_tokens_B": (r["training_progress"]["tokens_seen"] or 0) / 1e9})
    (dest / "comparison_100k.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["## 相同 100,000 步的固定验证", "", "各组训练量均为 **1.6384B**；固定验证量 **0.000065536B**。Δ 表示当前方案减去同 H baseline，负数更好。", ""]
    for scope in ("fixed", "full"):
        if scope == "full":
            lines += ["", "## 准确 100,000 步的最终全量验证", "", "验证量 **0.001048576B**。H4 Both 的最终权重为 147,609 步，不纳入本表。", ""]
        lines += ["| H | 方案 | Loss ↓ | PPL ↓ | ΔLoss | PPL 相对变化 |", "|---:|---|---:|---:|---:|---:|"]
        for row in output[scope + "_validation"]:
            label = LABELS[row["mode"]] + ("（历史 100k）" if row["historical_100k_record_of_overbudget_run"] else "")
            lines.append(f"| {row['H']} | {label} | {row['loss']:.6f} | {row['PPL']:.5f} | {row['delta_loss']:+.6f} | {row['PPL_change_percent']:+.4f}% |")
    lines += ["", "## 同 100k 固定验证上的逐 loop FFN 稀疏度", "", "以下均为各 loop 内 6 层的等权均值；不是替代逐层数据。阈值 ε=0.01，近零比例越高表示该阈值下更多分支输出接近零，不保证语言模型质量更好。", "", "| H | 方案 | loop | FFN 近零比例 | Top-10% 能量占比 |", "|---:|---|---:|---:|---:|"]
    for row in output["ffn_loop_means_fixed_100k"]:
        lines.append(f"| {row['H']} | {LABELS[row['mode']]} | {row['loop']} | {row['mean_near_zero_eps0.01']:.4%} | {row['mean_top10_energy']:.4%} |")
    (dest / "tables_100k.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"fixed_100k_rows": len(output["fixed_validation"]), "full_100k_rows": len(output["full_validation"]), "output_dir": str(dest)}))


if __name__ == "__main__":
    main()
