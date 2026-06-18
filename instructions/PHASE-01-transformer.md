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
- `src/train_single_gpu.py` – Training loop with AdamW, cross-entropy loss, checkpoint saving every N steps.
- `configs/phase1_124M.yaml` – Hyperparameters: `n_layer=12, n_head=12, n_embd=768, batch_size=32, lr=3e-4`.
- `slurm/train_1gpu.sh` – Single-node, single-GPU SLURM script that runs `train_single_gpu.py`.

---

## 3. Acceptance Criteria
- [ ] `model.py` forward pass runs without shape errors for a dummy batch (`bs=4, seq=1024`).
- [ ] Single-batch overfit test: loss drops to < 0.01 within 100 steps (validates backprop).
- [ ] Full training on TinyStories for 1 hour produces coherent (though broken) text completions.
- [ ] Checkpoints are saved to `checkpoints/run_{timestamp}/` with `model.pt`, `config.yaml`, and `optimizer.pt`.
- [ ] Training script accepts `--config` argument to load YAML hyperparameters.

---

## 4. Amarel Edge Case (Must Handle)
- Use `torch.cuda.amp` (autocast) to save memory. If an A100 is available, use `torch.compile` for speed (fallback gracefully if not).
- Store datasets in `$SCRATCH` (e.g., `$SCRATCH/data/tinystories/`) and symlink or pass path via config.
- Set `torch.manual_seed(42)` and `torch.backends.cudnn.deterministic = True` for reproducibility.

---

## 5. Cursor Instructions
- Follow `@GLOBAL.md`: type hints, pure functions, no global state.
- Use `omegaconf` to load YAML configs. Hard-coded numbers are forbidden except `vocab_size=50257`.
- Do not use HuggingFace `Trainer` or `transformers` for the model—only for tokenizer fallback if `tiktoken` is unavailable.
- Validate gradient flow by logging gradient norms every 100 steps (`torch.nn.utils.clip_grad_norm_`). 