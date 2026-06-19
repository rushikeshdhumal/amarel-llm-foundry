#!/bin/bash
#SBATCH --job-name=hello_1n1g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL

# ── Environment ───────────────────────────────────────────────────────────────
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# ── Resolve master address from SLURM_NODELIST ────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))")
export MASTER_ADDR
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun \
  --standalone \
  --nproc_per_node=1 \
  src/dist_hello.py \
  --backend nccl
