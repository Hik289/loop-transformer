#!/usr/bin/env python3
"""Audit the completed 52-condition screen and save reproducible numeric evidence."""
from pathlib import Path
import argparse
import json
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.sparsedrop_export import validate_final_diagnostics
from aggregate_sparsedrop import load_run, pairing_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    runs = [load_run(p) for p in sorted(args.results.iterdir())
            if (p / "summary.json").is_file() and (p / ".source_host.json").is_file()]
    assert len(runs) == 52
    baselines = {r["condition"]["loops"]: r for r in runs if r["condition"]["mode"] == "none"}
    records = []
    for r in runs:
        c, s, e = r["config"], r["summary"], r["evaluation"]
        h = r["condition"]["loops"]
        d = e["diagnostics"]
        acts = validate_final_diagnostics(d, loops=h, num_layers=6, hidden_size=768)
        assert e["tokens"] == 1048576 and e["chunks"] == 2048
        assert r["seed"] == 42 and c["arguments"]["data_seed"] == 100045
        pair_errors = pairing_errors(baselines[h], r) if r["complete"] else ["historical overbudget"]
        if r["complete"]:
            assert not pair_errors, (r["directory"], pair_errors)
            assert s["step"] == 100000 and s["tokens_seen"] == 1638400000
        def activation_means(rows):
            assert len({(a["tokens"], a["hidden_size"]) for a in rows}) == 1
            return {
                "near_zero": {str(t): statistics.mean(next(v["near_zero_ratio"] for v in a["threshold_sensitivity"] if v["threshold"] == t) for a in rows) for t in (0.001, 0.01, 0.1)},
                "residual_rms": statistics.mean(a["residual_rms"] for a in rows),
                "top_10pct_energy": statistics.mean(a["top_10pct_energy"] for a in rows),
            }
        records.append({
            "name": Path(r["directory"]).name, **r["condition"], "strict_100k": r["complete"],
            "steps": s["actual_steps"], "training_tokens_B": s["actual_tokens_seen"] / 1e9,
            "loss": e["loss"], "ppl": e["perplexity"],
            "ppl_improvement_vs_baseline_pct": 100 * (1 - e["perplexity"] / baselines[h]["evaluation"]["perplexity"]) if r["complete"] else None,
            "train_flops_estimate": s["approx_train_flops"],
            "mean_update_cosine": d["cross_loop"].get("mean_update_cosine"),
            "mean_state_cosine": d["cross_loop"].get("mean_state_cosine"),
            "mean_linear_cka": d["cross_loop"].get("mean_linear_cka"),
            "diagnostic_position_sha256": d["protocol"]["sample_position_sha256"],
            "sparsity_rows": len(acts), "pairing_errors": pair_errors,
            "branches": {branch: activation_means([a for a in acts if a["branch"] == branch]) for branch in ("attention", "ffn")},
            "per_loop": {str(loop): {branch: activation_means([a for a in acts if a["branch"] == branch and a["loop"] == loop]) for branch in ("attention", "ffn")} for loop in range(1, h + 1)},
        })
    valid = [r for r in records if r["strict_100k"]]
    assert len(valid) == 51
    assert len({r["diagnostic_position_sha256"] for r in records}) == 1
    fixed = [r for r in valid if r["schedule"] == "constant" and r["mode"] != "none"]
    lookup = {(r["loops"], r["mode"], r["p"], r["schedule"], r["delta"]): r for r in valid}
    pairs = []
    for mode in ("attention", "ffn", "both"):
        for delta in (0.05, 0.1):
            dec = lookup[(8, mode, 0.1, "decreasing", delta)]
            inc = lookup[(8, mode, 0.1, "increasing", delta)]
            pairs.append({"mode": mode, "delta": delta, "decreasing_ppl": dec["ppl"], "increasing_ppl": inc["ppl"], "decreasing_better": dec["ppl"] < inc["ppl"]})
    summary = {
        "total_ended": len(records), "strict_100k": len(valid), "historical_overbudget": 1,
        "total_final_sparsity_rows": sum(r["sparsity_rows"] for r in records),
        "strict_final_sparsity_rows": sum(r["sparsity_rows"] for r in valid),
        "total_effective_training_tokens_B": sum(r["summary"]["actual_tokens_seen"] for r in runs) / 1e9,
        "strict_effective_training_tokens_B": sum(r["summary"]["actual_tokens_seen"] for r in runs if r["complete"]) / 1e9,
        "fixed_dropout_wins_by_H": {str(h): {"wins": sum(r["ppl_improvement_vs_baseline_pct"] > 0 for r in fixed if r["loops"] == h), "total": sum(r["loops"] == h for r in fixed)} for h in (1, 2, 4, 8)},
        "ranking": [r["name"] for r in sorted(valid, key=lambda r: r["ppl"])],
        "schedule_direction_pairs": pairs,
    }
    assert summary["total_final_sparsity_rows"] == 2952
    output = {"summary": summary, "runs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
