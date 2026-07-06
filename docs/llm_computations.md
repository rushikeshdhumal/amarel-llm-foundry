# LLM Math & Computation — Quick Reference

Formulas and numbers used across Phases 1–3 of this project, written so you can follow the math without prior HPC or LLM training background.

For runtime bugs (OOM, NCCL hangs, FSDP deadlocks), see `torch_notes.md`.

---

## Key terms (read this first)

| Term | Meaning |
| :--- | :--- |
| **Token** | A piece of text the model reads/writes (e.g. `" the"`, `" dog"`). GPT-2 uses ~50k tokens (**vocab_size**). |
| **Parameter (param)** | One learnable number in the network. A **124M** model has ~124 million params. |
| **n_layer (L)** | Number of stacked **TransformerBlock** layers (model depth). |
| **n_head (H)** | Number of parallel **attention heads** per layer. Each head learns different relationships between tokens. |
| **n_embd (D)** | **Embedding dimension** — width of the vector representing each token (must divide evenly by **n_head**). |
| **head_dim** | `n_embd / n_head` — size of each attention head's subspace. |
| **seq_len (T)** | **Context length** — max tokens the model sees at once (1024 in this project). |
| **batch_size (B)** | Number of **sequences processed in parallel on one GPU** (not total across all GPUs). |
| **world_size (N)** | Total number of **GPU processes** in the job (1 GPU → N=1; 4 GPUs → N=4; 4 nodes × 4 GPUs → N=16). |
| **rank** | Index of one GPU process (0 … N−1). **Rank 0** usually prints logs and saves checkpoints. |
| **Step** | One **optimizer update**: forward pass → backward pass → weight update on one mini-batch. |
| **Activation** | Intermediate tensor saved during **forward** so **backward** can compute gradients (uses VRAM). |
| **Gradient** | Direction each param should move to reduce **loss**; computed in **backward pass**. |
| **Loss** | Single number measuring prediction error (lower = better). We use **cross-entropy loss**. |
| **Perplexity** | `exp(loss)` — average "branching factor" at each token; easier to interpret than raw loss. |
| **Learning rate (lr)** | Step size for weight updates. Too high → unstable; too low → slow convergence. |
| **Warmup** | LR starts at 0 and ramps up linearly for the first **warmup_steps** (stabilises early training). |
| **fp32 / fp16** | 32-bit vs 16-bit floating point. **fp16** is half the memory and often faster; **fp32** is more precise. |
| **AdamW** | Optimizer that adapts per-parameter step sizes and applies **weight decay** (L2 regularisation). |
| **VRAM** | GPU memory. Training fails with **OOM** (out of memory) when VRAM is exceeded. |
| **DDP** | **DistributedDataParallel** — full model copy on every GPU; syncs gradients via **all-reduce**. |
| **FSDP** | **FullyShardedDataParallel** — splits params/grads/optimizer across GPUs (**sharding**). |
| **All-gather** | Collective where each GPU sends its shard and receives everyone else's (rebuilds full tensor briefly). |
| **tok/s** | **Tokens per second** — throughput; how much text the cluster processes per second. |

**Notation in formulas:** B = batch_size, T = seq_len, N = world_size, L = n_layer, H = n_head, D = n_embd.

---

## 1. GPT parameter count

**What we're counting:** every weight the model learns. More params → more capacity to fit patterns, but more GPU memory and compute.

**Per `TransformerBlock`** (one layer of the GPT-2 stack):

| Piece | What it does | Params (approx.) |
| :--- | :--- | ---: |
| **Attention** | Each token "looks at" past tokens (Q, K, V projections + output projection) | `4 × n_embd²` |
| **MLP** | Transforms each token independently (expand 4× then project back) | `8 × n_embd²` |

```
Subtotal per block ≈ 12 × n_embd²
```

**Full model:**

| Piece | What it does | Params |
| :--- | :--- | ---: |
| **Token embedding** | Maps token ID → vector of size **n_embd** | `vocab_size × n_embd` |
| **Positional embedding** | Adds "position in sequence" info (positions 0 … seq_len−1) | `seq_len × n_embd` |
| **TransformerBlocks** | Repeated L times | `n_layer × 12 × n_embd²` |
| **LM head** | Projects hidden state → logits over vocabulary | **Weight-tied** to token embedding (no extra params) |

```
Total ≈ n_layer × 12 × n_embd² + vocab_size × n_embd
```

**Weight tying:** the input embedding matrix and output projection share the same weights — halves embedding params and often improves quality.

| Phase | Config (L / H / D) | Params |
| :--- | :--- | ---: |
| 1 | 12 / 12 / 768 | ~124M |
| 2 | 24 / 16 / 1024 | ~354M |
| 3 | 24 / 16 / 2048 | ~1.31B |

**Constraint:** `n_embd % n_head == 0` — each head must get an equal slice: `head_dim = n_embd / n_head`.

---

## 2. GPU memory budget

Training memory = **static** (weights, optimizer, gradients) + **activations** (temporary forward tensors) + **communication buffers** + overhead.

### DDP (Phase 2) — full replica per GPU

**DDP** copies the **entire model** onto every GPU. Each rank stores:

| Component | Why it's stored | Bytes per param |
| :--- | :--- | ---: |
| **Weights** | Current model values | 4 (fp32) |
| **Gradients** | From backward pass, before optimizer step | 4 (fp32) |
| **AdamW m** | First moment (running average of gradients) | 4 (fp32) |
| **AdamW v** | Second moment (running average of squared gradients) | 4 (fp32) |

```
Static memory per GPU = n_params × 16 bytes
```

Example (350M, DDP): `354M × 16 B ≈ 5.5 GB` **per GPU**, before activations.

### FSDP (Phase 3) — sharded across `world_size`

**FSDP** splits each tensor across N GPUs. Each rank only **owns 1/N** of params, grads, and optimizer state at rest:

```
Static shard per rank = n_params × 14 bytes / world_size
  param shard (fp16)       n_params × 2 / N
  grad shard (fp32)        n_params × 4 / N
  AdamW m + v (fp32)       n_params × 8 / N
```

**All-gather peak:** during forward/backward, FSDP temporarily **reconstructs one full TransformerBlock** on each GPU to run the math, then discards it:

```
Peak ≈ (n_params / n_layer) × 2 bytes   (fp16 gathered weights for one block)
```

Example (1.3B, 4 GPUs): static shard `≈ 4.28 GB/rank`; all-gather peak `≈ 104 MB/block`.

**Observed on Amarel (1.3B FSDP, 4 ranks, bs=2):** ~5.5 GB `mem alloc` per rank (PyTorch-reported allocated VRAM).

---

## 3. Activation memory — the O(B × T²) trap

**Activations** are intermediate values kept from the forward pass so backward can compute gradients. They often dominate memory for long sequences.

**Self-attention** builds a score matrix: for each pair of token positions (i, j), "how much should token i attend to token j?" Shape: `(batch, n_head, T, T)`.

```
Attention per layer (fp16):  B × n_head × T × T × 2 bytes
All layers:                  n_layer × B × n_head × T² × 2

MLP + residuals per sample:  ≈ 6 × n_embd × T × 2 bytes
```

**Why T² matters:** doubling **seq_len** quadruples attention memory. Doubling **batch_size** also quadruples it (both appear in B × T²).

Example (12 layers, 12 heads, T=1024, fp16):

| batch_size | Attention memory |
| ---: | ---: |
| 64 | ~19 GB |
| 16 | ~4.7 GB |

**FlashAttention** (not used here yet) recomputes attention during backward instead of storing the full T×T matrix → O(T) memory instead of O(T²).

---

## 4. `auto_batch_size()` — runtime ceiling

**Problem:** HPC schedulers assign different GPU tiers (A100 40GB vs RTX 3090 24GB). A fixed config **batch_size** may OOM on smaller GPUs.

**Solution:** estimate safe batch size from actual VRAM at job start. Config value is a **ceiling**, not a target:

```
usable = (total_vram − static − comm_buffers − 500MB overhead) × 0.75
max_bs = usable / act_per_sample
safe_bs = min(config_bs, max_bs)
```

The **0.75 safety factor** covers scratch tensors and memory fragmentation not captured by the formula.

| Training mode | Static term | Extra comm term |
| :--- | :--- | :--- |
| **DDP** | `n_params × 16` (full replica) | NCCL all-reduce buffer ≈ `n_params × 1` |
| **FSDP** | `n_params × 14 / world_size` (shard) | One-block all-gather peak |

**All ranks must agree:** `dist.all_reduce(MIN)` — if one GPU is smaller, **every** rank drops to its safe batch size (otherwise NCCL collectives desynchronise and hang).

Example (1.3B FSDP, 4 GPUs):

| GPU | VRAM | `computed_max_bs` | Config | Used |
| :--- | ---: | ---: | ---: | ---: |
| RTX 3090 | 23.6 GB | 10 | 2 | 2 |
| A100-40GB | 39.5 GB | 19 | 2 | 2 |

---

## 5. Batch size & learning rate

### Effective batch size

**Per-GPU batch_size** is only part of the story. Total data per step:

```
tokens_per_step     = batch_size × world_size × seq_len
sequences_per_step  = batch_size × world_size
```

- **batch_size:** sequences on **one** GPU  
- **world_size:** number of GPUs (each processes its own mini-batch in parallel)  
- **seq_len:** tokens per sequence  

| Run | bs/GPU | GPUs | seq | Tokens/step |
| :--- | ---: | ---: | ---: | ---: |
| Phase 1 | 16 | 1 | 1024 | 16,384 |
| Phase 2 DDP | 4 | 4 | 1024 | 16,384 |
| Phase 3 FSDP (4-node) | 2 | 16 | 1024 | 32,768 |

### Linear LR scaling (Goyal et al., 2017)

**Intuition:** with N× more GPUs, each step averages gradients over N× more samples → gradient magnitude grows. Scale **learning rate** by N so each step's effective update size stays similar.

```
effective_lr = base_lr × world_size
```

- **base_lr:** value in the YAML config (before scaling in the training script)  
- **effective_lr:** what AdamW actually uses after the script multiplies by **world_size**

| Phase | base_lr | world_size | effective_lr |
| :--- | ---: | ---: | ---: |
| 1 | 3e-4 | 1 | 3e-4 |
| 2 | 3e-4 | 4 | 1.2e-3 |
| 3 | 6e-5 | 16 | 9.6e-4 |

### Cosine schedule with warmup

**Warmup** — LR ramps from ~0 to **max_lr** (avoids huge early updates while AdamW moments are cold):

```
lr = max_lr × (step + 1) / warmup_steps        (step < warmup_steps)
```

**Cosine decay** — after warmup, LR smoothly decreases to ~0 following a cosine curve:

```
progress = (step − warmup_steps) / (max_steps − warmup_steps)
lr = max_lr × 0.5 × (1 + cos(π × progress))    (step ≥ warmup_steps)
```

Check at step 100 with `max_lr=2.4e-4`, `warmup_steps=5000`:  
`lr = 2.4e-4 × 101/5000 ≈ 4.8e-6` ✓

---

## 6. Throughput & scaling

### Tokens per second (tok/s)

**Throughput** — how fast the cluster processes training data:

```
tok/s = log_interval × batch_size × world_size × seq_len / elapsed_seconds
```

Measured over the last **log_interval** steps (100 in this project). Higher tok/s = faster training.

**Ignore early steps** — first 1–2 log lines include CUDA kernel compilation and NCCL connection setup. Use steps **300+** for steady-state numbers.

### Multi-node scaling efficiency

**Ideal:** N× GPUs → N× throughput. **Reality:** communication (gradient sync, FSDP all-gather) adds overhead.

```
efficiency = (multi-node tok/s) / (1-node tok/s × node_count)
```

For Phase 3 (1 node → 4 nodes, 4× GPUs each): compare 4-node tok/s to `1-node tok/s × 4`. Target: **> 80%**.

**Match GPU tiers:** a 3090 baseline vs an A100 multi-node job gives meaningless efficiency. Always check which GPU you got:

```bash
grep "GPU:" gpt_fsdp_4nodes_<JOBID>.out | sort -u
```

| 1-node GPU | Steady tok/s (1.3B FSDP) | 80% target (4-node) |
| :--- | ---: | ---: |
| 4× RTX 3090 | ~5,750 | > 18,400 tok/s |
| 4× A100-40GB | ~8,250 | > 26,400 tok/s |

### Phase 2 DDP speedup (measured)

```
1 GPU (124M):   ~46,274 tok/s
4 GPU (124M):  ~165,074 tok/s
Speedup: 3.57×  (89% of ideal 4× — lost 11% to all-reduce communication)
```

---

## 7. Loss & perplexity

**Cross-entropy loss** measures how surprised the model is by the correct next token:

```
loss = −(1/N) Σ log P(correct_token | context)    (natural log, lower = better)
```

**Perplexity** converts loss to "average number of equally likely choices":

```
perplexity = exp(loss)
```

| Loss | Perplexity | Meaning |
| ---: | ---: | :--- |
| 1.03 | 2.8 | Phase 1 best (124M) — model narrows to ~3 plausible tokens on average |
| 0.72 | 2.1 | Phase 2 best (350M) |
| 6.99 | 1085 | Early training (step 100, LR still in warmup) — essentially random |

**Larger models should beat smaller ones** on the same data (lower loss), not merely match them.

**Batch loss vs logged loss:** training logs every 100 steps, but "save best" can fire on any step's mini-batch loss — small differences between logged and best-checkpoint loss are normal.

---

## 8. Mixed precision — bytes & roles

Training uses multiple precisions simultaneously:

| Component | Precision | Why |
| :--- | :--- | :--- |
| **Master weights** | fp32 | AdamW needs precision for small accumulated updates |
| **Forward compute** | fp16 | ~2× faster, half the activation memory |
| **FSDP all-gather** | fp16 | Half the bytes sent over the network between nodes |
| **Gradients** | fp16 (scaled) | **GradScaler** multiplies loss before backward to avoid underflow to zero |
| **Loss** | fp32 | Keeps the training signal numerically stable |

**FSDP:** use built-in `MixedPrecision` policy (handles gather/scatter in fp16). Do **not** also use `torch.autocast` — they overlap incorrectly.

**GradScaler is still required** with FSDP mixed precision — it solves gradient underflow, which FSDP's policy does not.

---

## 9. Checkpoint size (order of magnitude)

A **checkpoint** saves model (+ optionally optimizer) state so training can resume or run inference.

| Format | Size (1.3B model) | Notes |
| :--- | ---: | :--- |
| fp32 full **state dict** | ~5.2 GB | Gathers all params onto rank 0 — OOM risk with FSDP |
| **DCP sharded** (16 ranks) | ~330 MB/rank | Each GPU writes only its shard; no full-model gather |

---

## 10. Sanity checks before long runs

| Check | What it verifies | Pass criterion |
| :--- | :--- | :--- |
| **Overfit test** | Forward, backward, optimizer code paths work | Tiny 2-layer proxy model, loss < 0.01 in 100 steps |
| **FSDP sharding** | Memory split evenly across GPUs | All ranks show same `static_shard` and `mem alloc` |
| **LR at step 100** | Warmup schedule is active | `≈ max_lr × 100 / warmup_steps` |
| **Throughput baseline** | Steady tok/s before scaling comparison | Average tok/s from steps 300+ on 1 node |

---

## 11. Phase 4 evaluation metrics

Phase 4 measures checkpoints on a **fixed TinyStories validation memmap** (`val.bin`), pretokenized with the same BPE rules as training.

| Metric | Formula | Notes |
| :--- | :--- | :--- |
| **Val loss** | Mean cross-entropy over non-overlapping windows | Same `(x, y)` slicing as training; stride = `seq_len + 1` |
| **Perplexity** | `exp(val_loss)` | Average branching factor at each token; lower is better |
| **Bits per token** | `val_loss / ln(2)` | Information content in bits; comparable across models |

**Comparison caveats (required in benchmark report):**

- Project models: TinyStories pretrain. GPT-2 baselines: WebText pretrain.
- Same GPT-2 BPE tokenizer — tokenizer-fair, **not** training-data-fair.
- Phase 3 (~1.3B, 24L×2048D) vs `gpt2-xl` (~1.5B, 48L×1600D) is a **param-count approximate** comparison, not architecture-matched.
- GPT-2 may look worse on TinyStories val than in-domain project models — that reflects domain mismatch, not necessarily worse modeling.
