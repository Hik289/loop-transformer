#!/usr/bin/env bash
#SBATCH --job-name=prepare-fineweb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/prepare_fineweb_%j.out
#SBATCH --error=logs/prepare_fineweb_%j.err
#SBATCH --open-mode=append
set -euo pipefail
cd "${SPARSEDROP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source cluster/env.sh
# Dataset acquisition needs network access; tokenization uses 8 workers with
# four tokenizer threads each, within the 32 allocated CPUs. No GPU requested.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u scripts/prepare_fineweb_sparsedrop.py \
    --output-dir "$SPARSEDROP_ROOT/data/fineweb_edu20b" \
    --num-train-tokens 20000000001 \
    --num-validation-tokens 1048577 \
    --num-test-tokens 1048577 \
    --workers 8 \
    --tokenizer-threads 4 \
    --batch-size 256 \
    --split-seed 42
