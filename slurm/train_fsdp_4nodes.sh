#!/bin/bash
# train_fsdp_4nodes.sh — Multi-node, 4-node × 4-GPU FSDP training job (Phase 03).
#
# Submit from repo root on scratch (required for multi-node path visibility):
#   cd /scratch/$USER/amarel-llm-foundry
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
# MULTI-NODE RENDEZVOUS (critical):
#   sbatch runs this script on the FIRST node only. Calling torchrun directly
#   would launch ONE process group that waits forever for 3 other nodes →
#   RendezvousTimeoutError.
#   Fix: wrap torchrun in srun so it runs once per node (--ntasks-per-node=1).
#   Each copy uses SLURM_PROCID as --node_rank (0, 1, 2, 3).
#   MASTER_ADDR is the first node in SLURM_NODELIST, resolved to an IP (see below).

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
# Extract the first node hostname from SLURM_NODELIST and resolve to IP.
# Hostnames alone can fail cross-node DNS; IP is more reliable (Phase 0 fix).
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))")
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR:    $MASTER_ADDR"
echo "MASTER_PORT:    $MASTER_PORT"
echo "SLURM_NNODES:   $SLURM_NNODES"
echo "SLURM_NTASKS:   $SLURM_NTASKS"

# ── Ensure Python finds src/ on every node ────────────────────────────────────
# train_fsdp.py loads configs with relative paths (configs/base_config.yaml),
# so every rank must start in the repo root. --chdir pins the cwd per srun task;
# PYTHONPATH is a fallback if cwd propagation differs across node pools.
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export CONFIG TRAIN_DATA RESUME_FLAG

# ── Preflight: verify repo is visible on ALL allocated nodes ──────────────────
# Fails fast (before 16 GPUs spin up torchrun) if any node cannot see the repo.
# Requires submitting from the scratch clone:
#   cd /scratch/$USER/amarel-llm-foundry && sbatch slurm/train_fsdp_4nodes.sh
echo "── Preflight: verifying repo on all nodes ──"
if ! srun --chdir="$SLURM_SUBMIT_DIR" --label /bin/bash -c \
    'test -f src/train_fsdp.py && test -f configs/phase3_1B.yaml && echo "OK: $(hostname) $PWD"'; then
    echo "ERROR: repo not found at $SLURM_SUBMIT_DIR on one or more nodes."
    echo "Submit from: /scratch/$USER/amarel-llm-foundry"
    exit 1
fi
echo "Preflight: all nodes can access the repo."

# ── Launch FSDP training via srun + torchrun ──────────────────────────────────
# srun with --ntasks-per-node=1 runs this command ONCE PER NODE.
# SLURM_PROCID is 0 on node 0, 1 on node 1, … → correct --node_rank.
# torchrun then spawns --nproc_per_node=4 worker processes within each node.
#
# Works for single-node baselines too: sbatch --nodes=1 runs srun once on one node.
#
# --rdzv_backend=c10d: TCP rendezvous at MASTER_ADDR:MASTER_PORT (no etcd on Amarel).
#
# bash -c with single quotes: $SLURM_PROCID is expanded inside each srun task's
# shell (not once by the batch-script bash), giving each node a unique --node_rank.
srun --chdir="$SLURM_SUBMIT_DIR" --label /bin/bash -c '
exec torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m src.train_fsdp \
    --config "$CONFIG" \
    --data_path "$TRAIN_DATA" \
    $RESUME_FLAG
'
