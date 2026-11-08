#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop_check
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.err
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source cluster/env.sh
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -u scripts/sparsedrop_gpu_check.py
