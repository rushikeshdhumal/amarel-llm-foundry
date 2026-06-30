# PHASE 03: FSDP HPC Sharding (Multi-Node)

**Branch**: `feature/03-fsdp-hpc-sharding`  
**Prerequisite**: `feature/02-ddp-multi-gpu` (working DDP training on single node).  
**Goal**: Scale to multiple Amarel nodes using PyTorch's `FullyShardedDataParallel` (FSDP) to shard model parameters, gradients, and optimizer states across GPUs.

---

## 1. Objective

Replace `DistributedDataParallel` with FSDP. Enable model sharding across 4 nodes (16 GPUs total), integrate gradient checkpointing and FSDP-native mixed precision, and measure multi-node scaling efficiency.

**Why FSDP over DDP here:**  
DDP replicates the full model on every GPU — each rank holds all parameters, gradients, and AdamW states simultaneously. For a 1.3B model at fp32 that is `1.3B × 16 bytes ≈ 20 GB` per GPU, which exceeds a 40 GB A100's headroom once activations are added. FSDP shards all three across all ranks: with 16 GPUs each rank holds only `~1.3 GB` of parameters, making the model fit.

---

## 2. Files to Create / Modify

| File | Action | Notes |
| :--- | :--- | :--- |
| `src/train_fsdp.py` | Create | Start from `train_ddp.py`; replace DDP wrapper with FSDP + `size_based_auto_wrap_policy`; switch to FSDP-native mixed precision + `GradScaler`; use `torch.distributed.checkpoint` for saving |
| `src/model.py` | No changes needed | `TransformerBlock` is already a clean `nn.Module` — FSDP can wrap it directly; no structural changes required |
| `configs/phase3_1B.yaml` | Create | `n_layer=24, n_head=16, n_embd=2048` → **~1.31B params** (see §3 for the corrected parameter count) |
| `slurm/train_fsdp_4nodes.sh` | Create | Multi-node SLURM script: `--nodes=4 --ntasks-per-node=1` (see fix §F2 below) |

---

## 3. Model Architecture (`phase3_1B.yaml`)

```yaml
n_layer: 24      # same depth as Phase 2; doubling width rather than depth
n_head: 16       # head_dim = 2048 / 16 = 128  ✅  (must divide n_embd evenly)
n_embd: 2048     # 2× Phase 2's 1024
```

> **[FIX F1] `n_head=20` is invalid.**  
> 2048 ÷ 20 = 102.4 — not an integer. `CausalSelfAttention.__init__` asserts
> `n_embd % n_head == 0` and will crash immediately. Use `n_head=16` (head_dim=128).

**Corrected parameter count:**

```
Token embedding:    50257 × 2048           ≈  103M
Positional:          1024 × 2048           ≈    2M
Per TransformerBlock:
  Attention (Q,K,V,O): 4 × 2048²          ≈ 16.8M
  MLP (fc + proj):     8 × 2048²          ≈ 33.6M
  LayerNorms:                              ≈   0.0M
  subtotal per block                       ≈ 50.3M
24 blocks:          24 × 50.3M            ≈ 1,208M
Final LayerNorm + LM head (weight-tied):   ≈   0.1M
─────────────────────────────────────────────────────
Total                                      ≈ 1,313M  (~1.3B)
```

> **[FIX F5] The original config (`n_layer=32, n_embd=2048`) produces ~1.72B params, not ~1B.**  
> `n_layer=24, n_embd=2048` gives ~1.31B — a clean 4× parameter jump from Phase 2's 350M,
> with the same depth and doubled width. Document it as "~1.3B", not "~1B".

---

## 4. Acceptance Criteria

- [ ] Model sharding verified: `torch.cuda.memory_allocated()` per GPU is approximately
      `total_param_memory / world_size` (each rank holds ~1/16 of the parameters).
- [ ] **Multi-node scaling efficiency > 80%** when going from 1 node (4 GPUs, FSDP) to
      4 nodes (16 GPUs, FSDP), measured in tokens/second.  
      `efficiency = (4-node tok/s) / (1-node tok/s × 4)`.  
      *(Note: Amarel uses Ethernet, not InfiniBand. If efficiency falls below 80%,
      document the interconnect bottleneck in `docs/torch_notes.md` rather than treating
      it as a code bug.)*
- [ ] Gradient checkpointing applied via `apply_activation_checkpointing()` on
      `TransformerBlock` units after FSDP wrapping (see §5 for correct API).
- [ ] Sharded checkpoints saved via `torch.distributed.checkpoint` (DCP); never
      `torch.save(model.state_dict())` with default FSDP state dict type.
- [ ] **Training loss drops below Phase 2 best (0.72) within 10k steps**, confirming
      the 1.3B model is exploiting its extra capacity.

> **[FIX F6] Original criterion 5 said "within 0.1 of Phase 2 perplexity".**  
> A 1.3B model (4× larger than Phase 2's 350M) trained on the same data should converge
> to a *lower* loss than Phase 2's 0.72, not the same. "Within 0.1" would mean the larger
> model is barely better — that would indicate a bug, not success. The correct bar is
> "better than Phase 2 best".

---

## 5. Implementation Notes

### FSDP wrapping

Use `size_based_auto_wrap_policy` with a min-params threshold that causes FSDP to wrap each `TransformerBlock` individually:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial

wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000)
model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=local_rank)
```

### Mixed precision — use FSDP-native policy, keep `GradScaler`

> **[FIX F8] `FSDP MixedPrecision` and `GradScaler` serve different roles — both are needed.**

```python
from torch.distributed.fsdp import MixedPrecision
import torch

mp_policy = MixedPrecision(
    param_dtype=torch.float16,    # parameters gathered/scattered in fp16
    reduce_dtype=torch.float16,   # gradient all-reduce in fp16
    buffer_dtype=torch.float16,   # buffers (e.g. causal mask) in fp16
)
model = FSDP(model, mixed_precision=mp_policy, ...)
```

- **FSDP `MixedPrecision`** handles precision during the parameter gather/scatter
  operations that happen inside FSDP — `torch.autocast` is unaware of these and would
  leave them in fp32. Drop `torch.autocast` from the training loop.
- **`GradScaler`** is still required. FSDP's MP policy does not prevent fp16 gradient
  underflow; the scaler's loss-scaling is independent of how FSDP manages shards.

### `torch.compile` — compile per-block BEFORE FSDP wrapping

> **[FIX F4] The original instruction said "only after FSDP wrapping". This is wrong.**  
> For the same reason as DDP (documented in `train_ddp.py`): compiling after wrapping
> embeds FSDP's gather/scatter hooks into the compiled graph. TorchDynamo does not
> understand these collectives and retraces on every step. Compile the raw module first.

```python
# compile individual blocks BEFORE FSDP wraps them
if use_compile:
    for i, block in enumerate(model.blocks):
        model.blocks[i] = torch.compile(block)

model = FSDP(model, ...)
```

> **Note:** `use_compile: false` in `phase3_1B.yaml` until RHEL9 GCC version on Amarel
> GPU nodes is confirmed compatible with TorchInductor. Do not change this default
> without first running the Phase 0 smoke test with `torch.compile` enabled.

### Gradient checkpointing — correct API

> **[FIX F7] `activation_checkpointing_policy` is not a real PyTorch API.**

Apply **after** FSDP wrapping (this is the correct order for activation checkpointing
specifically — it wraps the already-FSDP'd `TransformerBlock` modules):

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from functools import partial
from src.model import TransformerBlock

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=lambda m: isinstance(m, TransformerBlock),
)
```

`NO_REENTRANT` is preferred over the default reentrant mode in PyTorch >= 2.0 because it
avoids issues with FSDP's own reentrant backward hooks.

### Sharded checkpointing

```python
import torch.distributed.checkpoint as dcp
from pathlib import Path

# Save (rank 0 coordinates, all ranks participate)
dcp.save({"model": model, "optimizer": optimizer}, checkpoint_id=str(ckpt_dir))

# Load (must call before FSDP wrapping; load into sharded model)
dcp.load({"model": model, "optimizer": optimizer}, checkpoint_id=str(ckpt_dir))
```

Never call `torch.save(model.state_dict())` with default FSDP state dict type — it gathers
the full model on rank 0 and OOMs on large models. Use DCP which writes one shard file per
rank and avoids the gather entirely.

### Per-rank memory logging

Log every 100 steps so GPU imbalance is visible early:

```python
if step % 100 == 0 and step > 0:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved  = torch.cuda.memory_reserved(device)  / 1024**3
    print(f"[rank {rank}] step {step} | mem alloc {allocated:.2f} GB | reserved {reserved:.2f} GB")
```

---

## 6. Amarel Edge Cases

### Cross-node NCCL interface

> **[FIX F3] Do NOT set `NCCL_SOCKET_IFNAME=eth0`.**  
> Amarel GPU nodes use unpredictable Ethernet interface names (`ens`, `enp`, `bond`, etc.).
> Hardcoding `eth0` means NCCL finds no matching interface and falls back to loopback,
> killing all cross-node bandwidth.

`common.sh` already sets the correct value:
```bash
export NCCL_SOCKET_IFNAME=^lo,^docker   # exclude loopback + docker; auto-select the rest
export NCCL_IB_DISABLE=1                # Amarel does not have IB between GPU nodes
export NCCL_TIMEOUT=1800                # 30-min timeout; FSDP all-gathers are larger than DDP
```

No changes needed to `common.sh` for Phase 3.

### SLURM script — `ntasks-per-node` must be 1

> **[FIX F2] `--ntasks-per-node=4` launches 4 `torchrun` processes per node.**  
> Each tries to bind the same rendezvous port → `EADDRINUSE` crash. We hit this exact
> bug in Phase 0 (`hello_2nodes_4gpu.sh`) and fixed it. The correct SLURM pattern is:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # ONE torchrun launcher per node
#SBATCH --cpus-per-task=8          # 2 CPUs per GPU × 4 GPUs
#SBATCH --gres=gpu:4               # 4 GPUs per node → 16 total
```

`torchrun --nproc_per_node=4` then spawns the 4 GPU workers internally on each node.

### Checkpoint path

Save sharded checkpoints to `$SCRATCH/checkpoints/` using DCP. The `--resume` flag in
`train_fsdp.py` should load via `dcp.load()`. Do not use a `--save-sharded` flag
(non-standard) — DCP is always sharded by default.

### Timeout

`NCCL_TIMEOUT=1800` (already in `common.sh`) is sufficient for 1.3B FSDP all-gathers
on Ethernet. Increase to `3600` only if jobs are dying with NCCL timeout errors during
the parameter gather phase.

---

## 7. Key Differences from Phase 2 (DDP)

| Aspect | Phase 2 DDP | Phase 3 FSDP |
| :--- | :--- | :--- |
| Model replication | Full copy per GPU | Sharded (1/world_size per GPU) |
| Gradient sync | All-reduce after backward | Reduce-scatter during backward + all-gather before forward |
| Optimizer states | Full copy per GPU | Sharded (ZeRO-3 equivalent) |
| Mixed precision | `torch.autocast` + `GradScaler` | FSDP `MixedPrecision` + `GradScaler` |
| Checkpointing | `torch.save(model.module.state_dict())` | `torch.distributed.checkpoint.save()` |
| `torch.compile` | Before DDP wrap | Before FSDP wrap (per-block) |
| Gradient checkpointing | Not needed (350M fits easily) | After FSDP wrap, via `apply_activation_checkpointing` |
| SLURM nodes | 1 | 4 |
| Total GPUs | 4 | 16 |
