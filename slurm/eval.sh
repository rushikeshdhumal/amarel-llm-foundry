#!/bin/bash
# eval.sh — Single-GPU: eval all project checkpoints, then run GPT-2 benchmark report.
#
# Prerequisites:
#   1. sbatch slurm/pretokenize_val.sh
#   2. Edit configs/eval_checkpoints.yaml with real checkpoint paths
#   3. sbatch slurm/export_checkpoint.sh (Phase 3 only, if needed)
#
# Submit:
#   sbatch slurm/eval.sh

#SBATCH --job-name=gpt_eval
#SBATCH --partition=gpu-redhat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
mkdir -p "$HF_HOME"

VAL_PATH="$SCRATCH/data/tinystories/tokenized/val.bin"
EVAL_RESULTS="$SCRATCH/eval_results"
CHECKPOINTS_CONFIG="configs/eval_checkpoints.yaml"

if [ ! -f "$VAL_PATH" ]; then
    echo "ERROR: $VAL_PATH not found. Run: sbatch slurm/pretokenize_val.sh"
    exit 1
fi

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Val data:    $VAL_PATH"
echo "HF cache:    $HF_HOME"

mkdir -p "$EVAL_RESULTS"

# ── Parse checkpoint list from YAML (name + path pairs) ───────────────────────
mapfile -t CKPT_LINES < <(
    python3 - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/eval_checkpoints.yaml")
for entry in cfg.checkpoints:
    path = os.path.expandvars(str(entry.path))
    if "<" in path:
        continue
    print(f"{entry.name}\t{path}")
PY
)

if [ "${#CKPT_LINES[@]}" -eq 0 ]; then
    echo "WARNING: No valid checkpoints in $CHECKPOINTS_CONFIG (placeholders still present?)"
    echo "Edit configs/eval_checkpoints.yaml with real paths before running."
fi

# ── Eval each project checkpoint ──────────────────────────────────────────────
for line in "${CKPT_LINES[@]}"; do
    NAME="${line%%$'\t'*}"
    CKPT_PATH="${line#*$'\t'}"

    if [ ! -f "$CKPT_PATH/model.pt" ]; then
        echo "Skipping $NAME — model.pt not found at $CKPT_PATH"
        continue
    fi

    echo "── Evaluating $NAME ──"
    python -m src.eval \
        --checkpoint "$CKPT_PATH" \
        --val_path "$VAL_PATH" \
        --name "$NAME" \
        --output_dir "$EVAL_RESULTS" \
        --batch_size 1
done

# ── Benchmark report (project JSON + GPT-2 baselines) ───────────────────────
echo "── Running benchmark ──"
python -m src.benchmark \
    --config configs/eval_suite.yaml \
    --checkpoints_config "$CHECKPOINTS_CONFIG" \
    --batch_size 1

echo "Done. Results: $EVAL_RESULTS/  Report: $SCRATCH/benchmark_results/report.md"
