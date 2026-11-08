#!/usr/bin/env bash
#SBATCH --job-name=sparsedrop
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/400m_%A_%a.out
#SBATCH --error=logs/400m_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source cluster/env.sh
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STUDY="${SPARSEDROP_STUDY:-400m}"
LAUNCH_INFO=$(python scripts/select_experiment.py "$STUDY" "$SLURM_ARRAY_TASK_ID")
mapfile -t LAUNCH <<< "$LAUNCH_INFO"
CONFIG="${LAUNCH[0]}"
OUTPUT_DIR="${LAUNCH[1]}"
if [[ "${LAUNCH[2]}" == complete ]]; then
    python -c 'import sys; from pathlib import Path; from src.training.sparsedrop_export import export_final_sparsity; export_final_sparsity(Path(sys.argv[1]))' "$OUTPUT_DIR"
    if [[ "$STUDY" == 400m ]]; then python -u scripts/compact_sparsedrop_400m_checkpoints.py "$OUTPUT_DIR"; fi
    echo "Already completed the selected experiment budget; training skipped."
    exit 0
fi
ARRAY_ELEMENT="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ) job=$SLURM_JOB_ID array=$ARRAY_ELEMENT restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) config=$CONFIG"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))'
python -u scripts/long_training_gpu_preflight.py > "logs/400m_exact_budget_${ARRAY_ELEMENT}_r${SLURM_RESTART_COUNT:-0}.out"

if [[ "$STUDY" == 400m ]]; then
python -u scripts/preflight_sparsedrop_400m.py --tmp-root /tmp \
  --report "provenance/400m_preflight_${ARRAY_ELEMENT}_r${SLURM_RESTART_COUNT:-0}.json" \
  > "logs/400m_preflight_${ARRAY_ELEMENT}_r${SLURM_RESTART_COUNT:-0}.out"
fi

GPU_MONITOR_PID=""
TRAIN_PID=""
WAIT_INTERRUPTED=0
CANCEL_STATUS=0

cleanup_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}

forward_signal() {
    local signal_name=$1
    WAIT_INTERRUPTED=1
    if [[ -n "$TRAIN_PID" ]]; then
        kill -"$signal_name" "$TRAIN_PID" 2>/dev/null || true
    fi
}

# Traps only forward the signal. The guarded wait below collects the actual
# trainer status after Bash's signal-interrupted wait returns.
trap cleanup_monitor EXIT
trap 'forward_signal USR1' USR1
trap 'CANCEL_STATUS=143; forward_signal TERM' TERM
trap 'CANCEL_STATUS=130; forward_signal INT' INT

nvidia-smi --id="${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv --loop=30 > "logs/400m_gpu_${ARRAY_ELEMENT}_r${SLURM_RESTART_COUNT:-0}.csv" &
GPU_MONITOR_PID=$!
python -u scripts/train_sparsedrop.py --config "$CONFIG" --max-seconds 601200 &
TRAIN_PID=$!

while true; do
    WAIT_INTERRUPTED=0
    if wait "$TRAIN_PID"; then
        TRAIN_STATUS=0
    else
        TRAIN_STATUS=$?
    fi
    if (( WAIT_INTERRUPTED )); then
        continue
    fi
    break
done
TRAIN_PID=""
cleanup_monitor

# A user cancellation must stay cancelled even though the trainer checkpoints
# and returns 99 when it receives TERM/INT.
if (( CANCEL_STATUS != 0 )); then
    exit "$CANCEL_STATUS"
fi

if (( TRAIN_STATUS == 99 )); then
    echo "REQUEUE $(date -u +%Y-%m-%dT%H:%M:%SZ) array=$ARRAY_ELEMENT after safe checkpoint"
    # The trainer and GPU monitor are finished before returning this exact
    # array element to Slurm, so the original array's %2 cap stays in force.
    trap - USR1 TERM INT
    if scontrol requeue "$ARRAY_ELEMENT"; then
        exit 0
    else
        REQUEUE_STATUS=$?
        echo "ERROR: requeue failed for $ARRAY_ELEMENT (exit $REQUEUE_STATUS); checkpoint retained" >&2
        exit "$REQUEUE_STATUS"
    fi
fi

if (( TRAIN_STATUS != 0 )); then
    echo "ERROR: trainer failed with exit $TRAIN_STATUS; not requeueing" >&2
    exit "$TRAIN_STATUS"
fi
if [[ "$STUDY" == 400m ]]; then python -u scripts/compact_sparsedrop_400m_checkpoints.py "$OUTPUT_DIR"; fi
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ) job=$SLURM_JOB_ID array=$ARRAY_ELEMENT"
