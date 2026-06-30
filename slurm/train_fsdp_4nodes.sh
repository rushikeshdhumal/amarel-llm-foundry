#!/bin/bash
# train_fsdp_4nodes.sh — Multi-node, 4-node × 4-GPU FSDP training job (Phase 03).
#
# Submit from repo root:
#   sbatch slurm/train_fsdp_4nodes.sh
#
# Resume from a sharded DCP checkpoint:
#   sbatch slurm/train_fsdp_4nodes.sh --resume $CHECKPOINT_DIR/run_<ts>/step_<N>
#
# Baseline comparison (1 node, 4 GPUs, same config — for scaling efficiency check):
#   sbatch --nodes=1 slurm/train_fsdp_4nodes.sh --config configs/phase3_1B.yaml
#
# HOW THIS DIFFERS FROM train_ddp_4gpu.sh
# ─────────────────────────────────────────
# train_ddp_4gpu.sh:   1 node,  4 GPUs,  DDP (full model replica per GPU)
# train_fsdp_4nodes.sh: 4 nodes, 16 GPUs, FSDP (sharded across all GPUs)
#
# KEY SLURM DIFFERENCES:
#   --nodes=4:            4 nodes (vs 1 for Phase 2)
#   --ntasks-per-node=1:  ONE torchrun launcher per node (NOT 4).
#                         torchrun spawns nproc_per_node workers internally.
#                         Setting ntasks-per-node=4 would launch 4 torchrun
#                         processes all trying to bind the same port → EADDRINUSE.
#                         (Same bug fixed in Phase 0 hello_2nodes_4gpu.sh.)
#
# MULTI-NODE RENDEZVOUS:
#   torchrun uses MASTER_ADDR / MASTER_PORT for the initial rendezvous.
#   MASTER_ADDR is set to the first node in SLURM_NODELIST (see below).
#   All 4 nodes connect to that address before any collective runs.

#SBATCH --job-name=gpt_fsdp_4nodes
#SBATCH --partition=gpu-redhat
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # ONE torchrun launcher per node; spawns 4 workers
#SBATCH --cpus-per-task=8          # 2 CPUs per GPU × 4 GPUs
#SBATCH --gres=gpu:4               # 4 GPUs per node → 16 total
#SBATCH --mem=80G                  # 1.3B model shards + activations + DataLoader buffers
#SBATCH --time=24:00:00            # 50k steps at 1.3B params on 4-node A100 ≈ 8–16h
#SBATCH --output=%x_%j.out         # stdout → gpt_fsdp_4nodes_<JOBID>.out
#SBATCH --error=%x_%j.err          # stderr → gpt_fsdp_4nodes_<JOBID>.err
#SBATCH --export=ALL

# ── Environment ───────────────────────────────────────────────────────────────
# common.sh loads cuda/12.8.1, activates the llm conda env, exports $SCRATCH,
# and sets NCCL variables (TORCH_NCCL_ASYNC_ERROR_HANDLING, NCCL_SOCKET_IFNAME, …).
# No changes to common.sh are needed for Phase 3.
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# ── Verify dataset before requesting 16 GPUs ─────────────────────────────────
TRAIN_DATA="$SCRATCH/data/tinystories/TinyStoriesV2-GPT4-train.txt"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Download with:"
    echo "  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt \\"
    echo "       -P $SCRATCH/data/tinystories/"
    exit 1
fi

# ── Print job context for debugging ───────────────────────────────────────────
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NODELIST"
echo "GPUs per node:  $CUDA_VISIBLE_DEVICES"
echo "Total GPUs:     $((SLURM_NNODES * 4))"
echo "Train data:     $TRAIN_DATA"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Submit dir:     $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR" || exit 1

# ── Parse optional flags ──────────────────────────────────────────────────────
CONFIG="configs/phase3_1B.yaml"
RESUME_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)  RESUME_FLAG="--resume $2"; echo "Resuming from: $2"; shift 2 ;;
        --config)  CONFIG="$2"; echo "Using config: $2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

# ── Multi-node rendezvous setup ───────────────────────────────────────────────
# Extract the first node hostname from SLURM_NODELIST.
# SLURM_NODELIST format examples: "hal[001-004]" or "hal001,hal002,hal003,hal004".
# scontrol show hostnames converts both to a newline-separated list.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "MASTER_ADDR:    $MASTER_ADDR"
echo "MASTER_PORT:    $MASTER_PORT"

export MASTER_ADDR
export MASTER_PORT

# ── Launch FSDP training via torchrun ─────────────────────────────────────────
# --nnodes=$SLURM_NNODES:         number of nodes (4)
# --nproc_per_node=4:             4 GPU workers per node
# --node_rank=$SLURM_NODEID:      which node this task is (0, 1, 2, 3)
# --master_addr / --master_port:  rendezvous point (node 0)
# --rdzv_backend=c10d:            use PyTorch's c10d rendezvous (TCP-based).
#   This is the correct backend for multi-node jobs with a known MASTER_ADDR.
#   The alternative "etcd" requires a separate etcd service and is not available
#   on Amarel.
# -m src.train_fsdp:              module invocation from repo root
torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m src.train_fsdp \
    --config "$CONFIG" \
    --data_path "$TRAIN_DATA" \
    $RESUME_FLAG
