# PHASE 01: NanoGPT Transformer (Single-GPU)

**Branch**: `feature/01-nanogpt-transformer`  
**Prerequisite**: `feature/00-hpc-distributed-baseline` (merge or rebase to inherit SLURM utilities).  
**Goal**: Implement a GPT-2 style decoder-only transformer from scratch and train it on a single Amarel GPU.

---

## 1. Objective
Build the core transformer modules (Attention, MLP, LayerNorm) in pure PyTorch. Train a ~124M parameter model on TinyStories or Shakespeare until it overfits a single batch, then run a full training loop on a small dataset.

---

## 2. Files to Create / Modify
- `src/model.py` – Contains `CausalAttention`, `MLP`, `TransformerBlock`, `GPT` (stack blocks + LM head).
- `src/tokenizer.py` – Wrapper for `tiktoken` (GPT-2 encoding). Hardcode `n_vocab = 50257`.
- `src/data_utils.py` – `IterableDataset` that streams text from `.txt` files, tokenizes, and chunks to `seq_len=1024`.
- `src/train_single_gpu.py` – Training loop with AdamW, cross-entropy loss, gradient clipping, checkpoint saving every N steps. Must accept `--config` and `--resume` flags.
- `configs/phase1_124M.yaml` – Hyperparameters: `n_layer=12, n_head=12, n_embd=768, batch_size=64, lr=3e-4, grad_clip=1.0, grad_norm_log_interval=100`.
- `slurm/train_1gpu.sh` – Single-node, single-GPU SLURM script that runs `train_single_gpu.py`.

---

## 3. Acceptance Criteria
- [ ] `model.py` forward pass runs without shape errors for a dummy batch (`bs=4, seq=1024`).
- [ ] Single-batch overfit test: loss drops to < 0.01 within 100 steps (validates backprop).
- [ ] Full training on TinyStories for 1 hour produces coherent (though broken) text completions.
- [ ] Checkpoints are saved to `checkpoints/run_{timestamp}/` with `model.pt`, `config.yaml`, and `optimizer.pt`.
- [ ] Training script accepts `--config` argument to load YAML hyperparameters.
- [ ] Training script accepts `--resume <checkpoint_dir>` flag and correctly restores model, optimizer, and step count.
- [ ] Gradient norms are logged every `grad_norm_log_interval` steps.

---

## 4. Amarel Edge Case (Must Handle)
- Use `torch.cuda.amp` (autocast) to save memory. Enable `torch.compile` when `torch.cuda.get_device_capability() >= (8, 0)` (Ampere/Ada Lovelace and above — the cluster provides NVIDIA L40S, sm_89, which qualifies). Fallback gracefully if compile fails.
- Store datasets in `$SCRATCH` (e.g., `$SCRATCH/data/tinystories/`). Assert the path exists at job start and fail fast with a clear message rather than crashing mid-training.
- Set `torch.manual_seed(42)` and `torch.backends.cudnn.deterministic = True` for reproducibility.

---

## 5. Cursor Instructions
- Follow `@GLOBAL.md`: type hints, pure functions, no global state.
- Use `omegaconf` to load YAML configs. Hard-coded numbers are forbidden except `vocab_size=50257`.
- Do not use HuggingFace `Trainer` or `transformers` for the model—only for tokenizer fallback if `tiktoken` is unavailable.
- Validate gradient flow by logging gradient norms every `grad_norm_log_interval` steps using `torch.nn.utils.clip_grad_norm_` (which returns the norm before clipping).
- Apply gradient clipping every step using `grad_clip` from config (see `@GLOBAL.md`).