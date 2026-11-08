#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop-stage-gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/stage_gate_%j.out
#SBATCH --error=logs/stage_gate_%j.err
#SBATCH --open-mode=append
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Intentionally no --gres or GPU request. Supply a checked-in JSON gate plan
# as the first positional argument. An optional local afterok dependency can
# postpone CPU allocation; this script independently validates both clusters.
if [[ $# -ne 1 ]]; then
    printf '%s\n' 'Usage: sbatch [--dependency=afterok:...] cluster/cross_cluster_stage_gate.slurm PLAN.json' >&2
    exit 2
fi
exec venv/bin/python scripts/cross_cluster_stage_gate.py \
  --plan "$1" --poll-seconds 300 --max-seconds 603900
