#!/bin/bash
# train_1gpu.sh — Single-GPU GPT-2 124M training job for Phase 01.
#
# Submit from repo root: sbatch slurm/train_1gpu.sh
# Resume:                sbatch slurm/train_1gpu.sh --resume checkpoints/run_<ts>/step_<N>

#SBATCH --job-name=gpt_1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4       # 4 CPUs: 2 for DataLoader workers + 2 headroom
#SBATCH --gres=gpu:1
#SBATCH --mem=48G               # model + activations + data buffers; L40S node has 256G+
#SBATCH --time=04:00:00         # 4-hour wall-clock limit; extend if needed
#SBATCH --output=%x_%j.out      # stdout → gpt_1gpu_<JOBID>.out
#SBATCH --error=%x_%j.err       # stderr → gpt_1gpu_<JOBID>.err
#SBATCH --export=ALL

# ── Environment ───────────────────────────────────────────────────────────────
# common.sh loads cuda/12.8.1, activates the llm conda env, exports $SCRATCH,
# and sets all NCCL variables. Must be sourced before any Python call.
source "$(dirname "$0")/common.sh"

# ── Verify dataset exists before requesting GPU time ─────────────────────────
# Fail fast here rather than 30 minutes into a job when the DataLoader crashes.
TRAIN_DATA="$SCRATCH/data/tinystories/TinyStoriesV2-GPT4-train.txt"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Download with:"
    echo "  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt \\"
    echo "       -P $SCRATCH/data/tinystories/"
    exit 1
fi

# ── Print job context for debugging ──────────────────────────────────────────
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Train data:    $TRAIN_DATA"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Submit dir:    $SLURM_SUBMIT_DIR"

# ── Change to repo root ───────────────────────────────────────────────────────
# All Python imports (src.*) are relative to the repo root.
cd "$SLURM_SUBMIT_DIR" || exit 1

# ── Optional: parse --resume argument passed to sbatch ───────────────────────
# Usage: sbatch slurm/train_1gpu.sh --resume checkpoints/run_20240601/step_0005000
RESUME_FLAG=""
if [ "$1" = "--resume" ] && [ -n "$2" ]; then
    RESUME_FLAG="--resume $2"
    echo "Resuming from: $2"
fi

# ── Launch training ───────────────────────────────────────────────────────────
# Single GPU = no torchrun needed; just call Python directly.
# --data_path passed explicitly so the config's ??? placeholder is filled.
python -m src.train_single_gpu \
    --config configs/phase1_124M.yaml \
    --data_path "$TRAIN_DATA" \
    $RESUME_FLAG
