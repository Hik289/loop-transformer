#!/usr/bin/env python3
"""Plot recorded loop similarity, layer RMS, and layer energy at exactly 100k."""
from pathlib import Path
import argparse
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_sparsedrop_layer_profiles import GROUPS, STYLES, layer_curve, read_runs


DIVERSITY = GROUPS[1]["series"]
LAYER_SERIES = GROUPS[2]["series"]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#A4AAB3", "axes.linewidth": .7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, output, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=190, facecolor="white")
    plt.close(fig)


def heading(fig, title, subtitle, handles, labels):
    fig.suptitle(title, x=.085, y=.982, ha="left", fontsize=19,
                 fontweight="bold", color="#172333")
    fig.text(.085, .937, subtitle, ha="left", fontsize=12, color="#526070")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.079, .904),
               ncol=2, frameon=False, fontsize=11, handlelength=2.8,
               columnspacing=2.7, labelspacing=.65)


def label(run, name):
    return f"{name}  |  PPL {run['ppl']:.3f}"


def plot_diversity(runs, output):
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.1), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, top=.765, bottom=.09, hspace=.43)
    handles, labels = [], []
    measures = [("state_cosine", "Output after each loop"),
                ("update_cosine", "Change made by each loop")]
    display_names = ["Baseline", "FFN: constant 10%", "FFN: decreasing 20% to 0%", "FFN: increasing 0% to 20%"]
    for ax, (metric, panel_title) in zip(axes, measures):
        for name, display, style in DIVERSITY:
            run = runs[name]
            y = run["relative_to_loop1"][metric]
            line, = ax.plot(np.arange(1, 9), y, linewidth=2.1, markersize=5.5,
                            markeredgecolor="white", markeredgewidth=.7,
                            **STYLES[style])
            if metric == "state_cosine":
                handles.append(line)
        labels = display_names
        ax.set_title(panel_title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(-.5, 1.05)
        ax.set_yticks([-.5, 0, .5, 1])
        ax.set_ylabel("Similarity to Loop 1\n(cosine)")
        ax.grid(axis="y", color="#D8DFE7", lw=.6)
        ax.axhline(0, color="#94A3B8", lw=.9, zorder=1)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelbottom=True, length=3, color="#A4AAB3")
        ax.set_xticks(np.arange(1, 9))
        ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=11)
        ax.set_xlim(.85, 8.15)
    axes[-1].set_xlabel("Loop number", labelpad=9)
    heading(fig, "Loop Diversity",
            "Each model is compared with its own Loop 1; lower values mean less similarity",
            handles, labels)
    save(fig, output, "loop_diversity_reference_loop1_h8")


def plot_layer_metric(runs, output, metric):
    is_rms = metric == "rms"
    title = "The best FFN schedule starts with much smaller outputs" if is_rms else "Output Concentration"
    subtitle = ("Eight loops | RMS measures output magnitude; legend shows validation PPL" if is_rms else
                "Percentage of squared output accounted for by the largest 10% of values")
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.1), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, top=.765, bottom=.09, hspace=.36)
    handles, labels = [], []
    for ax, branch in zip(axes, ("ffn", "attention")):
        for loop in range(8):
            left, right = loop * 6 + .5, (loop + 1) * 6 + .5
            if loop % 2 == 0:
                ax.axvspan(left, right, color="#EEF2F7", alpha=.75, zorder=0)
            if loop:
                ax.axvline(left, color="#BBC4CE", lw=.8, zorder=1)
            ax.text(loop * 6 + 3.5, 1.025, f"Loop {loop+1}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, color="#526070")
        maximum = 0
        for name, display, style in LAYER_SERIES:
            run = runs[name]
            points = layer_curve(run, branch)
            x = [p["position"] for p in points]
            y = [p[metric] * (1 if is_rms else 100) for p in points]
            maximum = max(maximum, max(y))
            line, = ax.plot(x, y, linewidth=1.9, markersize=3.6,
                            markeredgewidth=.6, markeredgecolor="white", zorder=3, **STYLES[style])
            if branch == "ffn":
                handles.append(line)
                labels.append(label(run, display))
        ax.set_ylim(0, maximum * 1.12 if is_rms else 100)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(.5, 48.5)
        branch_label = "FFN" if branch == "ffn" else "Attention"
        ax.set_ylabel(f"{branch_label}\nOutput RMS" if is_rms else f"{branch_label}\nLargest 10% share (%)")
        ax.grid(axis="y", color="#D8DFE7", lw=.6, alpha=.8, zorder=0)
        ax.set_xticks(np.arange(1, 49))
        ax.set_xticklabels([str(i % 6 + 1) for i in range(48)], fontsize=9)
        ax.tick_params(axis="x", length=2.5, color="#A4AAB3", labelbottom=True)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Shared layer within each loop (the same six layers are reused)", labelpad=9)
    heading(fig, title, subtitle, handles, labels)
    save(fig, output, "layer_output_rms_h8" if is_rms else "layer_top10_output_share_h8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    project = parser.parse_args().project_root
    output = project / "figures/sparsedrop_additional"
    output.mkdir(parents=True, exist_ok=True)
    all_runs = read_runs(project)
    chosen = sorted({s[0] for s in DIVERSITY + LAYER_SERIES})
    runs = {name: all_runs[name] for name in chosen}
    for name, run in runs.items():
        source = project / run["source"]
        record = next(r for line in source.read_text().splitlines()
                      if (r := json.loads(line)).get("step") == 100000 and r.get("event") == "validation_fixed")
        cross = record["diagnostics"]["cross_loop"]
        assert run["H"] == 8 and cross["loop_indices"] == list(range(1, 9))
        run["adjacent_loops"] = {}
        run["relative_to_loop1"] = {}
        for metric in ("update_cosine", "state_cosine", "linear_cka"):
            a = np.array(cross[metric])
            assert a.shape == (8, 8) and np.isfinite(a).all()
            assert np.allclose(a, a.T, atol=1e-12) and np.allclose(a.diagonal(), 1)
            assert a.min() >= (-1 if "cosine" in metric else 0) - 1e-10 and a.max() <= 1 + 1e-10
            run["adjacent_loops"][metric] = [float(a[i, i-1]) for i in range(1, 8)]
            run["relative_to_loop1"][metric] = [float(a[i, 0]) for i in range(8)]
        run["cross_loop"] = cross
        run["loop_states"] = record["diagnostics"]["loop_states"]
        run["branch_loop_means"] = {}
        for branch in ("ffn", "attention"):
            points = layer_curve(run, branch)
            run["branch_loop_means"][branch] = [
                {"loop": loop,
                 "rms": statistics.mean(p["rms"] for p in points if p["loop"] == loop),
                 "top10_energy_percent": statistics.mean(p["top10_energy"] * 100 for p in points if p["loop"] == loop)}
                for loop in range(1, 9)]
    baseline = runs["h8_none_p0.00_constant_d0.00_s42"]
    for run in runs.values():
        run["branch_vs_baseline"] = {}
        for branch in ("ffn", "attention"):
            pairs = list(zip(layer_curve(run, branch), layer_curve(baseline, branch)))
            run["branch_vs_baseline"][branch] = {
                "positions": len(pairs),
                "rms_lower": sum(p["rms"] < b["rms"] for p, b in pairs),
                "top10_energy_lower": sum(p["top10_energy"] < b["top10_energy"] for p, b in pairs),
                "top10_energy_higher": sum(p["top10_energy"] > b["top10_energy"] for p, b in pairs)}
    payload = {"step": 100000, "training_tokens_B": 1.6384, "diagnostic_tokens": 512,
               "validation_tokens_B": .000065536,
               "diversity_resolution": "loop boundaries only; each loop compared with Loop 1 of the same model",
               "layer_resolution": "eight loops, six shared layers, two residual branches",
               "top10_definition": "per token: energy of 77 largest-magnitude values / energy of all 768 values; then mean over 512 tokens",
               "diversity_series": DIVERSITY, "layer_series": LAYER_SERIES, "runs": runs}
    (output / "additional_profiles.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    setup()
    plot_diversity(runs, output)
    plot_layer_metric(runs, output, "rms")
    plot_layer_metric(runs, output, "top10_energy")
    print(json.dumps({"figures": 3, "representative_settings": len(runs),
                      "all_source_runs_checked": len(all_runs), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
