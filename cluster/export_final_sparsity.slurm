#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop-sparsity-export
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sparsity_export_%j.out
#SBATCH --error=logs/sparsity_export_%j.err
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Only parse existing JSON summaries; no model, checkpoint, or GPU is loaded.
venv/bin/python scripts/export_sparsedrop_sparsity.py results_fineweb20b --strict
