#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-51%2
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --signal=B:USR1@120
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source cluster/env.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=$(python -c 'import json,os; print(json.load(open("configs/sparsedrop/manifest_seed42.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])]["config"])')
echo "START $(date --iso-8601=seconds) job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID host=$(hostname) config=$CONFIG"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print("torch",torch.__version__,"cuda",torch.version.cuda,"device",torch.cuda.get_device_name(0))'
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv --loop=30 > "logs/gpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
python -u scripts/train_sparsedrop.py --config "$CONFIG" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true; wait "$TRAIN_PID"; exit 0' USR1
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null || true; wait "$TRAIN_PID"; exit 143' TERM INT
wait "$TRAIN_PID"
echo "END $(date --iso-8601=seconds) job=$SLURM_JOB_ID"
