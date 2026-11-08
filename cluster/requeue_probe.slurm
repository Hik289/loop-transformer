#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop-requeue-check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --requeue
#SBATCH --array=0-0%1
#SBATCH --output=logs/requeue_probe_%A_%a.out
#SBATCH --open-mode=append
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
printf 'restart=%s job=%s_%s\n' "${SLURM_RESTART_COUNT:-0}" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
if [[ ${SLURM_RESTART_COUNT:-0} == 0 ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    echo REQUEUE_CONFIRMED
fi
