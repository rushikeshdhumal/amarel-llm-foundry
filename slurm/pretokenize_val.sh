#!/bin/bash
# pretokenize_val.sh — CPU job: download TinyStories valid split (if missing) and build val.bin.
#
# Submit:
#   sbatch slurm/pretokenize_val.sh

#SBATCH --job-name=pretokenize_val
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL

source "$SLURM_SUBMIT_DIR/slurm/common.sh"

VALID_DIR="$SCRATCH/data/tinystories"
VALID_FILE="$VALID_DIR/TinyStoriesV2-GPT4-valid.txt"
VALID_URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

mkdir -p "$VALID_DIR"

if [ ! -f "$VALID_FILE" ]; then
    echo "Downloading TinyStories validation split..."
    wget -q "$VALID_URL" -O "$VALID_FILE"
    if [ ! -f "$VALID_FILE" ]; then
        echo "ERROR: Failed to download $VALID_URL"
        exit 1
    fi
    echo "Downloaded → $VALID_FILE"
else
    echo "Validation file already present: $VALID_FILE"
fi

cd "$SLURM_SUBMIT_DIR" || exit 1

python scripts/pretokenize_val.py \
    --input "$VALID_FILE" \
    --output_dir "$SCRATCH/data/tinystories/tokenized"
