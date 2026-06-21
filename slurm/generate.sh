#!/bin/bash
# generate.sh — Run inference on a trained GPT checkpoint.
#
# Submit:
#   sbatch slurm/generate.sh --checkpoint $SCRATCH/checkpoints/run_<ts>/step_0091500
#
# Or interactively (no sbatch queue wait):
#   srun --partition=gpu --gres=gpu:1 --mem=8G --time=00:10:00 --pty bash
#   cd $SLURM_SUBMIT_DIR && source slurm/common.sh
#   python -m src.generate --checkpoint $SCRATCH/checkpoints/run_<ts>/step_0091500
#
# Optional flags (all have defaults — only --checkpoint is required):
#   --prompts "Once upon a time" "There was a dog"   # one or more text prompts
#   --max_new_tokens 200                             # tokens to generate per prompt
#   --temperature 0.8                                # <1 = conservative, >1 = creative
#   --top_k 40                                       # restrict to top-40 tokens

#SBATCH --job-name=gpt_generate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2       # inference is single-threaded; 2 CPUs is plenty
#SBATCH --gres=gpu:1
#SBATCH --mem=8G                # 124M model at fp32 ≈ 500 MB; 8G gives comfortable headroom
#SBATCH --time=00:15:00         # inference is fast; 15 minutes is generous
#SBATCH --output=%x_%j.out      # stdout → gpt_generate_<JOBID>.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL

# ── Environment ───────────────────────────────────────────────────────────────
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# ── Require --checkpoint argument ─────────────────────────────────────────────
# Parse --checkpoint <path> from arguments passed to sbatch.
CHECKPOINT_PATH=""
MAX_NEW_TOKENS=150
TEMPERATURE=0.8
TOP_K=40

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top_k) TOP_K="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: --checkpoint <path> is required."
    echo "Example: sbatch slurm/generate.sh --checkpoint \$SCRATCH/checkpoints/run_<ts>/step_0091500"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH/model.pt" ]; then
    echo "ERROR: model.pt not found in $CHECKPOINT_PATH"
    echo "Available checkpoints under \$SCRATCH/checkpoints/:"
    ls "$SCRATCH/checkpoints/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# ── Job context ───────────────────────────────────────────────────────────────
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:  $CHECKPOINT_PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1

# ── Run inference ─────────────────────────────────────────────────────────────
# Default prompts are TinyStories-style story openings — a good match for the
# training distribution. Override with --prompts "..." in the sbatch call.
python -m src.generate \
    --checkpoint "$CHECKPOINT_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K"
