#!/bin/bash
# Amarel shared environment for all SLURM scripts

# Module loads
# Use absolute paths to initialise the module system and CUDA in SLURM batch
# jobs, where /etc/profile.d is not automatically sourced.
# shellcheck source=/dev/null
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /projects/community/modulefiles
module load cuda/12.8.1

# Activate the project conda environment.
# We source the conda init script with its absolute path rather than relying
# on `module load anaconda` because the module function may not be available
# in non-interactive SLURM batch scripts.
CONDA_BASE=/projects/community/anaconda/2025.06/ts840
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llm

# Force unbuffered Python output so print() lines appear immediately in .out
export PYTHONUNBUFFERED=1

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

# PyTorch memory and distributed settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# Create directories
mkdir -p $DATA_DIR $CHECKPOINT_DIR $LOG_DIR

echo "Amarel environment ready. SCRATCH=$SCRATCH"