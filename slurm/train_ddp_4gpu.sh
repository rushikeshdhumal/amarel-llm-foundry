#!/bin/bash
# train_ddp_4gpu.sh — Single-node, 4-GPU DDP training job for Phase 02.
#
# Submit from repo root:
#   sbatch slurm/train_ddp_4gpu.sh
#
# Resume from a checkpoint:
#   sbatch slurm/train_ddp_4gpu.sh --resume $CHECKPOINT_DIR/run_<ts>/step_<N>
#
# Baseline comparison (124M model, same config as Phase 1):
#   sbatch slurm/train_ddp_4gpu.sh --config configs/phase1_124M.yaml
#
# HOW THIS DIFFERS FROM train_1gpu.sh
# ─────────────────────────────────────
# train_1gpu.sh:   python -m src.train_single_gpu (one process, one GPU)
# train_ddp_4gpu:  torchrun --nproc_per_node=4  (4 processes, 4 GPUs)
#
# torchrun is PyTorch's process launcher for DDP. It:
#   1. Spawns nproc_per_node independent Python processes.
#   2. Sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT in each.
#   3. Monitors all processes; if one crashes it kills the others immediately.
#
# On a single node, torchrun sets MASTER_ADDR=127.0.0.1 automatically.
# No manual NCCL rendezvous configuration is needed.

#SBATCH --job-name=gpt_ddp_4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # ONE torchrun launcher; it spawns 4 workers internally
#SBATCH --cpus-per-task=8          # 2 CPUs per GPU × 4 GPUs = 8; for DataLoader workers
#SBATCH --gres=gpu:4               # request all 4 GPUs on the node
#SBATCH --mem=80G                  # 350M model × 4 ranks + activations + DataLoader buffers
#SBATCH --time=24:00:00            # 100k steps at 350M params on 4× A100 ≈ 10–14h; 24h buffer
#SBATCH --output=%x_%j.out         # stdout → gpt_ddp_4gpu_<JOBID>.out
#SBATCH --error=%x_%j.err          # stderr → gpt_ddp_4gpu_<JOBID>.err
#SBATCH --export=ALL

# ── Environment ───────────────────────────────────────────────────────────────
# common.sh loads cuda/12.8.1, activates the llm conda env, exports $SCRATCH,
# and sets all NCCL variables (TORCH_NCCL_ASYNC_ERROR_HANDLING, NCCL_SOCKET_IFNAME, …).
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# ── Verify dataset exists before requesting 4 GPUs ────────────────────────────
# Failing here costs nothing; failing 30 min into a GPU job wastes allocation.
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

cd "$SLURM_SUBMIT_DIR" || exit 1

# ── Parse optional flags ──────────────────────────────────────────────────────
CONFIG="configs/phase2_350M.yaml"
RESUME_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)  RESUME_FLAG="--resume $2"; echo "Resuming from: $2"; shift 2 ;;
        --config)  CONFIG="$2"; echo "Using config: $2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

# ── Launch DDP training via torchrun ──────────────────────────────────────────
# --nnodes=1:           single-node run
# --nproc_per_node=4:   4 worker processes (one per GPU)
# --master_addr / --master_port: explicit for clarity; torchrun defaults to
#   these values on single-node anyway.
# -m src.train_ddp:     run the module (requires running from repo root, which
#   we cd'd to above, so Python finds src/ on sys.path).
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    -m src.train_ddp \
    --config "$CONFIG" \
    --data_path "$TRAIN_DATA" \
    $RESUME_FLAG
