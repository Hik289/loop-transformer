#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop-readme
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/readme_update_%j.out
#SBATCH --error=logs/readme_update_%j.err
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Catch up a process that loaded the trainer before the completion hook existed.
# This only reads JSON and writes Markdown; it requests no GPU.
venv/bin/python scripts/update_sparsedrop_readme.py \
  --manifest configs/sparsedrop_long/manifest.json \
  --results-root results_fineweb20b \
  --readme README_SparseDrop_Loop_Ouro_Metrics_Formulas_CN.md \
  --live-readme README_实验结果实时表_CN.md \
  --status-json analysis_fineweb20b/readme_table_status.json --strict
