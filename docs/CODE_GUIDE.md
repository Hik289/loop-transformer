# Code guide

## SparseLoop experiments

| Task | Code |
| --- | --- |
| Reuse the same layer stack across loops | `src/model/looplm.py` |
| Attention/FFN branch order and dropout | `src/model/transformer.py` |
| Model sizes and dropout schedules | `src/model/config.py` |
| Fixed-depth final-loop training | `scripts/train_sparsedrop.py` |
| FineWeb-Edu preparation | `scripts/prepare_fineweb_sparsedrop.py` |
| Small WikiText preparation utility | `scripts/prepare_sparsedrop_data.py` |
| Per-loop and per-layer measurements | `src/training/sparsedrop_metrics.py` |
| Validate and export final layer records | `src/training/sparsedrop_export.py`, `scripts/export_sparsedrop_sparsity.py` |
| Aggregate or compare training outputs | `scripts/aggregate_sparsedrop.py`, `scripts/report_sparsedrop_100k.py`, `scripts/analyze_sparsedrop_final.py` |
| Layer and loop figures | `scripts/plot_sparsedrop_layer_profiles.py`, `scripts/plot_sparsedrop_additional_profiles.py` |
| Main figure | `scripts/plot_overview.py` |
| GPU budget and model checks | `scripts/long_training_gpu_preflight.py`, `scripts/preflight_sparsedrop_400m.py` |
| Batch-size benchmarks | `scripts/benchmark_sparsedrop_400m_batch.py` |
| Completed checkpoint compaction | `scripts/compact_sparsedrop_400m_checkpoints.py` |

## Original LoopLM tools

The inherited `scripts/train.py`, `scripts/evaluate.py`, `scripts/eval_checkpoints.py`, and `scripts/finetune_gate.py` support the original training and exit-gate workflows. Arithmetic, CAPO, and MANO tasks are in `src/analysis/` and `scripts/analyze.py`. Their `.slurm` sweeps are included in `scripts/`. `src/inference/` contains caching, early exit, and an lm-evaluation-harness adapter. Optional MoE code remains in `src/model/moe_layer.py`; the reported SparseLoop studies use dense FFNs.

Install the optional dependencies with `pip install -e ".[legacy]"` before using these inherited tools. Their objectives and data differ from the SparseLoop experiments, so their results should not be mixed into the published dropout tables.

## Release scope

All current Python modules, experiment entry points, test files, and shell/Slurm scripts from the working codebase are included. The release contains the 52 small-model and 18 larger-model configurations actually used in this study. Unused multi-seed plans, machine environments, old source snapshots, administrative cleanup scripts, cluster job state, raw execution logs, datasets, and checkpoint weights are not part of the public package.

The release starts a new Git history. Configuration paths are relative, Slurm site settings are supplied by the user, and the public records contain scientific measurements instead of local machine metadata. Numerical model and training code is preserved; plotting paths and launch-path validation are adapted to the standalone repository layout.
