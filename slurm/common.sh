#!/bin/bash
# Amarel shared environment for all SLURM scripts

# Module loads
module use /projects/community/modulefiles
module load cuda/12.8.1
module load anaconda/2025.06-ts840
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm

# Scratch space (use for datasets & checkpoints)
export SCRATCH=/scratch/$USER
export DATA_DIR=$SCRATCH/data
export CHECKPOINT_DIR=$SCRATCH/checkpoints
export LOG_DIR=$SCRATCH/logs

# NCCL tuning for multi-node
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=1800

# PyTorch distributed
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# Create directories
mkdir -p $DATA_DIR $CHECKPOINT_DIR $LOG_DIR

echo "Amarel environment ready. SCRATCH=$SCRATCH"