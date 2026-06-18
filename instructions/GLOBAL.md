# GLOBAL RULES FOR CURSOR AGENT

## 1. System & Paths
- **Storage**: Always use `$SCRATCH` for datasets and checkpoints. Never write to `/home/`.
- **SLURM**: Source `slurm/common.sh` first. Use `torchrun` for multi-GPU, not `mpirun`. 
- **Environment**: Load modules via `module load cuda/12.1 python/3.10`.

## 2. PyTorch & Model
- **Framework**: Pure `torch.nn`. Never use HuggingFace `Trainer` or `pytorch-lightning`. 
- **Optimization**: Use `torch.compile` for single-GPU. Use `torch.cuda.amp` autocast for all multi-GPU runs.
- **Checkpointing**: Save `model.pt`, `optimizer.pt`, and `config.yaml` every 500 steps. Save to `checkpoints/run_{timestamp}/`.
- **Resume**: All training scripts must accept `--resume` flag.
- **Gradient clipping**: Always clip gradients with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Set `grad_clip` in config; never hard-code.

## 3. Data
- **Tokenizer**: Use `tiktoken` (GPT-2 encoding). Max sequence length = 1024.
- **Dataset**: Stream via `IterableDataset` or memory-mapped `.bin` files. Do not load full datasets into RAM.

## 4. Debugging & Safety
- **Overfit First**: Always validate by overfitting a single batch (loss must go to ~0) before full training.
- **NCCL**: Set `os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"` in every distributed script.
- **Determinism**: Use `torch.manual_seed(42)` globally. Disable benchmark mode during debugging.

## 5. Git & Code Quality
- **Branch Scope**: Only modify files explicitly listed in the phase instruction file.
- **Type Hints**: Mandatory for all function signatures.
- **Configs**: All hyperparameters go in `configs/*.yaml`. No hard-coded values in Python files.