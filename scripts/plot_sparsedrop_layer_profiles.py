#!/usr/bin/env python3
"""Plot saved 100k layer measurements; no model execution or cluster access."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.sparsedrop_export import validate_final_diagnostics


STYLES = {
    "baseline": dict(color="#374151", marker="o", linestyle="--"),
    "ffn_constant": dict(color="#2563EB", marker="s", linestyle="-"),
    "attention_constant": dict(color="#9333EA", marker="^", linestyle="-"),
    "both_constant": dict(color="#D97706", marker="D", linestyle="-"),
    "ffn_decreasing": dict(color="#00836D", marker="o", linestyle="-"),
    "ffn_increasing": dict(color="#C94357", marker="s", linestyle="-"),
    "both_decreasing": dict(color="#D97706", marker="D", linestyle="-"),
}


def name(h, mode, p=0.1, schedule="constant", delta=0.0):
    return f"h{h}_{mode}_p{p:.2f}_{schedule}_d{delta:.2f}_s42"


GROUPS = [
    {
        "file": "layer_sparsity_h4_placement", "H": 4,
        "title": "Where does dropout change the layer outputs?",
        "subtitle": "Four loops | Constant dropout p = 0.10, except baseline",
        "series": [
            (name(4, "none", 0), "Baseline", "baseline"),
            (name(4, "ffn"), "FFN only", "ffn_constant"),
            (name(4, "attention"), "Attention only", "attention_constant"),
            (name(4, "both"), "Both", "both_constant"),
        ],
    },
    {
        "file": "layer_sparsity_h8_ffn_schedule", "H": 8,
        "title": "FFN dropout mainly changes the early loops",
        "subtitle": "Eight loops | FFN schedules have the same mean dropout rate of 10%",
        "series": [
            (name(8, "none", 0), "Baseline", "baseline"),
            (name(8, "ffn"), "FFN: constant 10%", "ffn_constant"),
            (name(8, "ffn", schedule="decreasing", delta=.1), "FFN: 20% → 0%", "ffn_decreasing"),
            (name(8, "ffn", schedule="increasing", delta=.1), "FFN: 0% → 20%", "ffn_increasing"),
        ],
    },
    {
        "file": "layer_sparsity_h8_quality", "H": 8,
        "title": "More near-zero values do not guarantee better prediction",
        "subtitle": "Eight loops | Legend shows validation PPL; lower is better",
        "show_ppl": True,
        "series": [
            (name(8, "none", 0), "Baseline", "baseline"),
            (name(8, "ffn", schedule="decreasing", delta=.1), "FFN: 20% → 0%", "ffn_decreasing"),
            (name(8, "both", schedule="decreasing", delta=.1), "Both: 20% → 0%", "both_decreasing"),
            (name(8, "attention"), "Attention: constant 10%", "attention_constant"),
        ],
    },
]


def read_runs(project):
    runs = {}
    for directory in sorted((project / "results/80m").iterdir()):
        if not (directory / "snapshot.json").is_file():
            continue
        path = directory / "eval.jsonl"
        found = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "validation_fixed" and record.get("step") == 100000:
                found.append((record, line))
        assert len(found) == 1, directory.name
        record, line = found[0]
        config = json.loads((directory / "config.json").read_text())
        h = config["arguments"]["loops"]
        assert record["tokens_seen"] == 1638400000
        assert (record["tokens"], record["chunks"]) == (65536, 128)
        diagnostics = record["diagnostics"]
        acts = validate_final_diagnostics(diagnostics, loops=h, num_layers=6, hidden_size=768)
        points = []
        for row in acts:
            near = {str(v["threshold"]): v["near_zero_ratio"] * 100
                    for v in row["threshold_sensitivity"]}
            points.append({"loop": row["loop"], "layer": row["layer"] + 1,
                           "position": (row["loop"] - 1) * 6 + row["layer"] + 1,
                           "branch": row["branch"], "near_zero_percent": near,
                           "rms": row["residual_rms"], "top10_energy": row["top_10pct_energy"]})
        runs[directory.name] = {
            "H": h, "ppl": record["perplexity"], "step": 100000,
            "source": str(path.relative_to(project)),
            "source_record_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "protocol": diagnostics["protocol"], "points": points,
        }
    assert len(runs) == 52
    assert len({json.dumps(r["protocol"], sort_keys=True) for r in runs.values()}) == 1
    return runs


def layer_curve(run, branch):
    points = sorted((p for p in run["points"] if p["branch"] == branch), key=lambda p: p["position"])
    assert [p["position"] for p in points] == list(range(1, run["H"] * 6 + 1))
    return points


def make_figure(group, runs, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#A4AAB3", "axes.linewidth": .7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.1), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, top=.765, bottom=.09, hspace=.36)
    fig.suptitle(group["title"], x=.085, y=.982, ha="left", fontsize=19, fontweight="bold", color="#172333")
    fig.text(.085, .937, group["subtitle"], ha="left", fontsize=12, color="#526070")
    h = group["H"]
    handles, labels = [], []
    for ax, branch in zip(axes, ("ffn", "attention")):
        for loop in range(h):
            left, right = loop * 6 + .5, (loop + 1) * 6 + .5
            if loop % 2 == 0:
                ax.axvspan(left, right, color="#EEF2F7", alpha=.75, zorder=0)
            if loop:
                ax.axvline(left, color="#BBC4CE", lw=.8, zorder=1)
            ax.text(loop * 6 + 3.5, 1.025, f"Loop {loop + 1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, color="#526070")
        for run_name, label, style_key in group["series"]:
            run = runs[run_name]
            assert run["H"] == h
            points = layer_curve(run, branch)
            x = [v["position"] for v in points]
            y = [v["near_zero_percent"]["0.01"] for v in points]
            line, = ax.plot(x, y, linewidth=1.9, markersize=3.6,
                            markeredgewidth=.6, markeredgecolor="white", zorder=3, **STYLES[style_key])
            if branch == "ffn":
                handles.append(line)
                labels.append(f"{label}  |  PPL {run['ppl']:.3f}" if group.get("show_ppl") else label)
        maximum = max(p["near_zero_percent"]["0.01"] for n, _, _ in group["series"] for p in layer_curve(runs[n], branch))
        ax.set_ylim(0, maximum * 1.12)
        ax.set_xlim(.5, h * 6 + .5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#D8DFE7", lw=.6, alpha=.8, zorder=0)
        ax.set_ylabel(f"{'FFN' if branch == 'ffn' else 'Attention'}\nNear-zero outputs (%)", fontsize=11)
        ax.set_xticks(np.arange(1, h * 6 + 1))
        ax.set_xticklabels([str((i - 1) % 6 + 1) for i in range(1, h * 6 + 1)], fontsize=9)
        ax.tick_params(axis="x", length=2.5, color="#A4AAB3", labelbottom=True)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Shared layer within each loop (the same six layers are reused)", labelpad=9)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.079, .904),
               ncol=2, frameon=False, fontsize=11, handlelength=2.8, columnspacing=2.7, labelspacing=.65)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{group['file']}.{suffix}", dpi=190, facecolor="white")
    plt.close(fig)


def analyze(runs):
    reports = {}
    for name_, run in runs.items():
        branches = {}
        for branch in ("ffn", "attention"):
            points = layer_curve(run, branch)
            by_loop = []
            for loop in range(1, run["H"] + 1):
                subset = [p for p in points if p["loop"] == loop]
                by_loop.append({"loop": loop,
                                "near_zero_percent": statistics.mean(p["near_zero_percent"]["0.01"] for p in subset),
                                "rms": statistics.mean(p["rms"] for p in subset)})
            peak = max(points, key=lambda p: p["near_zero_percent"]["0.01"])
            branches[branch] = {"loop_means": by_loop, "peak": peak}
        reports[name_] = {"H": run["H"], "ppl": run["ppl"], **branches}
    multi = [r for r in reports.values() if r["H"] > 1]
    count = sum(r["ffn"]["loop_means"][-1]["near_zero_percent"] < r["ffn"]["loop_means"][0]["near_zero_percent"] for r in multi)
    assert len(multi) == count == 42
    return {"multiloop_settings": len(multi), "last_loop_ffn_below_first": count,
            "run_summaries": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    output = args.project_root / "figures/sparsedrop_layers"
    output.mkdir(parents=True, exist_ok=True)
    runs = read_runs(args.project_root)
    chosen = sorted({n for g in GROUPS for n, _, _ in g["series"]})
    data = {"threshold": .01, "step": 100000, "training_tokens_B": 1.6384,
            "validation_tokens_B": .000065536, "groups": GROUPS,
            "runs": {n: runs[n] for n in chosen}}
    (output / "layer_profiles.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "layer_analysis.json").write_text(json.dumps(analyze(runs), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for group in GROUPS:
        make_figure(group, runs, output)
    print(json.dumps({"figures": 3, "formats": ["png", "pdf", "svg"], "representative_settings": len(chosen),
                      "selected_layer_branch_records": sum(len(runs[n]["points"]) for n in chosen),
                      "all_runs_checked": len(runs), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
