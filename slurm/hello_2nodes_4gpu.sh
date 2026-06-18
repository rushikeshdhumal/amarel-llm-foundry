#!/bin/bash
#SBATCH --job-name=hello_2n4g
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL

# ── Environment ───────────────────────────────────────────────────────────────
source "$(dirname "$0")/common.sh"

# ── Resolve master address (first node in allocation) ────────────────────────
# scontrol expands compact notation like node[001-002] → node001\nnode002
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))")
export MASTER_ADDR
export MASTER_PORT=29500

# ── Cross-node NCCL tuning ────────────────────────────────────────────────────
# Amarel uses Ethernet (not InfiniBand) for inter-node traffic.
# NCCL_IB_DISABLE=1 and NCCL_SOCKET_IFNAME are already set in common.sh.
# Increase timeout for slow node bring-up on multi-node jobs.
export NCCL_TIMEOUT=1800

echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}  SLURM_NTASKS=${SLURM_NTASKS}"

# ── Launch via srun so each node gets its own torchrun process ────────────────
# torchrun --node_rank is derived from SLURM_PROCID injected by srun.
srun --label torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node=2 \
  --node_rank="${SLURM_PROCID}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  src/dist_hello.py \
  --backend nccl
