#!/bin/bash
# export_checkpoint.sh — Multi-node FSDP export: DCP shards → model.pt + config.yaml.
#
# Submit (default 4 nodes × 4 GPUs — match Phase 3 training):
#   sbatch slurm/export_checkpoint.sh --checkpoint $SCRATCH/checkpoints/run_<ts>/best
#
# 1-node baseline export (if training used sbatch --nodes=1):
#   sbatch --nodes=1 slurm/export_checkpoint.sh --checkpoint $SCRATCH/checkpoints/run_<ts>/best

#SBATCH --job-name=fsdp_export
#SBATCH --partition=gpu-redhat
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: --checkpoint <path> is required."
    echo "Example: sbatch slurm/export_checkpoint.sh --checkpoint \$SCRATCH/checkpoints/run_<ts>/best"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH/meta.yaml" ]; then
    echo "ERROR: meta.yaml not found in $CHECKPOINT_PATH"
    echo "Expected an FSDP DCP checkpoint directory (not an exported model.pt dir)."
    exit 1
fi

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NODELIST"
echo "Total GPUs:     $((SLURM_NNODES * 4))"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Export output:  ${CHECKPOINT_PATH}_export"

cd "$SLURM_SUBMIT_DIR" || exit 1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))")
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR:    $MASTER_ADDR"
echo "MASTER_PORT:    $MASTER_PORT"

srun --label torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=4 \
    --node_rank="${SLURM_PROCID}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/export_fsdp_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH"
