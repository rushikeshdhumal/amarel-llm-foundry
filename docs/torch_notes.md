# PyTorch Gotchas — Lessons from Phase 01

Practical issues hit during development, explained from first principles.

---

## 1. `import` inside a function shadows the module name

**What happened:**
```python
def train():
    torch.device("cuda")   # ← UnboundLocalError here
    ...
    import torch._dynamo   # ← this line, even though it's later, makes Python
                           #   treat 'torch' as a local variable throughout train()
```

**Why:** Python decides at compile time whether a name is local or global. If `torch` is assigned *anywhere* in the function (including via `import`), it's treated as local *everywhere* in that function — even in lines before the import.

**Fix:** Always put imports at the top of the file, never inside functions.

---

## 2. `torch.compile` fails *silently* on the first forward pass, not at the call site

**What happened:**
```python
try:
    model = torch.compile(model)   # ← no error here
except Exception:
    pass

_, loss = model(x, y)   # ← crash here: GCC too old for stdatomic.h
```

**Why:** `torch.compile` is *lazy*. It only compiles kernels when they are first executed, not when you call `torch.compile()`. So wrapping the compile call in `try/except` doesn't help — the error surfaces later.

**Fix:** Set `torch._dynamo.config.suppress_errors = True` *before* the first forward pass. This tells dynamo to fall back to eager (uncompiled) mode on any compilation failure instead of crashing.

---

## 3. Attention memory is O(B × T²) — batch size matters a lot

**What happened:** OOM with `bs=64`, `seq_len=1024`, 12 attention layers.

**Why:** During the forward pass, each attention layer computes a score matrix of shape `(B, n_head, T, T)`. These must be kept in GPU memory for the backward pass through softmax.

```
Memory for attention = n_layers × B × n_head × T × T × bytes_per_element
                     = 12 × 64 × 12 × 1024 × 1024 × 2  ≈  19 GB  (fp16)
                     = 12 × 16 × 12 × 1024 × 1024 × 2  ≈  4.7 GB (fp16)
```

Halving `T` or `B` quarters the attention memory. This is why FlashAttention (Phase 2+) is so important — it recomputes attention on-the-fly during backprop instead of storing the full matrix, reducing attention memory to O(T) instead of O(T²).

**Fix for now:** Use `bs=16`. Long-term: use FlashAttention.

---

## 4. Python stdout is buffered in SLURM jobs

**What happened:** A training job ran for hours with no output visible in `.out` file.

**Why:** When Python's stdout is not a terminal (e.g. redirected to a file in a SLURM job), it buffers output in large chunks. `print()` lines don't appear until the buffer fills or the process exits.

**Fix:** Set `PYTHONUNBUFFERED=1` in the SLURM script (done in `slurm/common.sh`). This forces every `print()` to flush immediately.

---

## 5. `conda activate` requires shell initialisation

**What happened:** `conda activate llm` gave `CondaError: Run 'conda init' before 'conda activate'` — even after `conda init` had been run.

**Why:** `conda init` modifies `~/.bashrc` to define the `conda` shell function. That change only takes effect in new shells or after `source ~/.bashrc`. The current shell session doesn't see it.

**Fix:** Run `source ~/.bashrc` after `conda init`, then `conda activate` works immediately without reconnecting.

---

## 6. `torch.cuda.set_device()` must be called *before* `init_process_group()`

**What happened:** A 2-node NCCL job failed immediately at the first `dist.barrier()` with `ncclInvalidUsage`. The `.err` file also warned: *"using GPU 0 to perform barrier as devices used by this process are currently unknown"*.

**Why:** `init_process_group()` builds the NCCL communicator. NCCL needs to know which physical GPU each rank owns at communicator-creation time — it uses that information to set up peer-to-peer paths between devices. Calling `set_device()` after the group is already initialised means NCCL built the communicator without that information, resulting in `ncclInvalidUsage` on the very first collective.

```python
# WRONG — device is unknown when the communicator is created
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)   # too late

# CORRECT — pin the GPU first, then create the communicator
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
```

**Fix:** Always `set_device(local_rank)` before `init_process_group()`. Also pass `device_id` explicitly (see next entry).

---

## 7. Pass `device_id` to `init_process_group()` in PyTorch 2.x

**What happened:** Even after fixing the ordering above, PyTorch still logged a warning that it was "guessing" the GPU for the barrier.

**Why:** PyTorch 2.x added a `device_id` parameter to `init_process_group()` so the framework can definitively associate a rank with its GPU from the start. Without it, PyTorch guesses — usually correctly, but the ambiguity can cause subtle hangs on some topologies.

```python
# PyTorch 2.x — pass device_id to eliminate the guessing warning
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    device_id=device,   # ← explicit GPU binding
)
```

**Fix:** Pass `device_id=torch.device(f"cuda:{local_rank}")` to `init_process_group()`. Requires PyTorch ≥ 2.0.

---

## 8. `dirname "$0"` is unreliable in SLURM batch jobs

**What happened:** `source "$(dirname "$0")/common.sh"` silently failed, leaving `$SCRATCH` unset.

**Why:** When you run `sbatch slurm/train.sh`, SLURM copies the script to a spool directory (e.g. `/var/spool/slurm/job123/slurm_script`). Inside the job, `$0` is that spool path — `dirname "$0"` points to the spool directory, not your repo, so `common.sh` is never found. The `source` fails silently.

**Fix:** Use `$SLURM_SUBMIT_DIR` instead — it always holds the directory where `sbatch` was called from:
```bash
source "$SLURM_SUBMIT_DIR/slurm/common.sh"
```

---

## 9. Overfit test batch size must match model capacity, not training config

**What happened:** A single-batch overfit test on a 124M model with `bs=4, seq=1024` ran for 100 steps at `lr=1e-3` and only reached loss 5.26 — reporting failure even though the model was correct (full training later converged to 1.03).

**Why:** Two compounding issues:
1. `seq=1024` means 4,096 prediction targets per batch. AdamW spends ~10 steps warming up its first and second moment estimates (`m`, `v`). With only 90 effective steps, there isn't enough time for a deep model to memorise thousands of targets.
2. `lr=1e-3` is the right LR for *generalisation*, not for *memorisation*. The overfit test doesn't need to generalise — it just needs to prove gradients flow.

**Second cause — dropout active in `model.train()` mode:**
With `dropout=0.1`, each forward pass randomly zeros 10% of activations — the model sees a *different effective network* every step on the same fixed batch. Loss oscillates rather than converges. Fix: `model.eval()` inside the test (disables dropout without affecting gradient computation).

**Third cause — `AdamW` default `weight_decay=0.01`:**
`torch.optim.AdamW` applies L2 weight decay by default. For memorisation, this actively fights convergence by penalising large weights — it creates a loss floor the optimiser can't go below. Symptom: monotonically decreasing but stalling loss with diminishing step sizes (e.g. 4.97 → 3.88 → 3.11 → 2.70 → 2.49). Fix: `weight_decay=0.0` in the overfit test optimizer.

**Fourth cause — the full 124M model is the wrong tool for a memorisation sanity check:**
Even with correct settings, the right `lr` for a 124M model is hard to find: `lr=0.01` is too slow (plateaus at loss ~3), `lr=0.1` overshoots through 12 transformer layers and diverges to loss ~24. The gradient amplification through depth makes the sweet spot model-size dependent.

**The correct fix — use a tiny proxy model for the test:**
The sanity check only needs to verify that the GPT *code* works (forward pass, loss, backward, optimizer). It does not need to use the full production model. A 2-layer, 64-dim model (~3.3M params) tests identical code paths but converges reliably at `lr=0.01` in 100 steps because gradient amplification through 2 layers is minimal.

```python
# FINAL — tiny proxy model: same code paths, predictable convergence
nano_cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, seq_len=32, dropout=0.0)
nano = GPT(nano_cfg).to(device)
nano.eval()
x = torch.randint(0, vocab_size, (4, 32), device=device)
y = torch.randint(0, vocab_size, (4, 32), device=device)
opt = torch.optim.AdamW(nano.parameters(), lr=0.01, weight_decay=0.0)
for step in range(100): ...
# Converges to < 0.01 reliably. The 124M model is not touched.
```

**Rule of thumb:** Never use the full production model for an overfit sanity check. Use the smallest model that exercises the same code paths.
