# Phase 3 — FSDP Multi-Node Sharding

**Branch**: `feature/03-fsdp-hpc-sharding`  
**Project**: [Amarel LLM Foundry](https://github.com/rushikeshdhumal/amarel-llm-foundry) — 5-phase journey from SLURM hello-world to scaled GPT training and reproducible evaluation on Amarel HPC.

---

## Goal

Scale to 4 Amarel nodes (16 GPUs) using PyTorch `FullyShardedDataParallel` (FSDP). Train a 1.3B-parameter GPT model that is too large for DDP (full-replica per GPU would require ~20 GB per GPU in fp32, leaving no room for activations on a 40 GB A100). FSDP shards parameters, gradients, and optimizer states across all ranks, reducing per-GPU static memory to ~1 GB.

---

## What was built

| File | Purpose |
| :--- | :--- |
| `src/train_fsdp.py` | FSDP training loop: NCCL setup, `size_based_auto_wrap_policy`, FSDP-native `MixedPrecision`, activation checkpointing, `torch.distributed.checkpoint` save/load |
| `configs/phase3_1B.yaml` | Phase overrides: `n_layer=24`, `n_head=16`, `n_embd=2048` (1.31B params) |
| `slurm/train_fsdp_4nodes.sh` | 4-node × 4-GPU SLURM job via `torchrun` with c10d rendezvous |

All model, data, and config utilities (`src/model.py`, `src/data_utils.py`, `src/train_single_gpu.py`) are shared from earlier phases — no duplication.

---

## Model architecture (1.3B)

```
Token embedding  (50257 × 2048)
Position embedding (1024 × 2048)
        ↓
 × 24  TransformerBlock              ← each wrapped as its own FSDP unit
        ├── LayerNorm
        ├── CausalSelfAttention  (16 heads, head_dim=128, causal mask)
        ├── LayerNorm
        └── MLP  (2048 → 8192 → 2048, GELU)
        ↓
LayerNorm → Linear (2048 → 50257, weight-tied to token embedding)
```

**Parameter count:**

```
Token embedding:     50257 × 2048  ≈  103M  (weight-tied with lm_head)
Positional:           1024 × 2048  ≈    2M
Per TransformerBlock:
  Attention (Q,K,V,O): 4 × 2048²  ≈ 16.8M
  MLP (fc + proj):     8 × 2048²  ≈ 33.6M
  subtotal                         ≈ 50.3M
24 blocks:           24 × 50.3M   ≈ 1,208M
Final LayerNorm + lm_head (tied):  ≈   0.1M
─────────────────────────────────────────────
Total                              ≈ 1,313M  (~1.3B)
```

---

## How FSDP training works

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     4-node Amarel job  (16 GPUs total)                       │
│                                                                              │
│  Node 0               Node 1               Node 2               Node 3      │
│  torchrun (4 workers) torchrun (4 workers) torchrun (4 workers) torchrun     │
│  Rank 0–3             Rank 4–7             Rank 8–11            Rank 12–15  │
│                                                                              │
│  Each rank holds 1/16 of parameters, gradients, and optimizer state:         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────────────────┐   │
│  │ shard 0  │  │ shard 1  │  │ shard 2  │  ...  │ shard 15             │   │
│  │ ~164 MB  │  │ ~164 MB  │  │ ~164 MB  │       │ ~164 MB              │   │
│  └──────────┘  └──────────┘  └──────────┘       └──────────────────────┘   │
│                                                                              │
│  FORWARD (per TransformerBlock):                                             │
│    1. All-gather → reconstruct full block weights (fp16, ~100 MB / block)    │
│    2. Run block forward on full weights                                      │
│    3. Discard gathered weights — keep only shard                             │
│                                                                              │
│  BACKWARD (per TransformerBlock, reverse):                                   │
│    1. All-gather → re-gather block weights for gradient computation          │
│    2. Run block backward                                                     │
│    3. Reduce-scatter → each rank accumulates its own gradient shard          │
│    4. Discard gathered weights                                               │
│                                                                              │
│  OPTIMIZER STEP:                                                             │
│    Each rank runs AdamW on its own parameter + gradient shard independently  │
│    No communication — all shards updated in parallel                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key design points:**
- `size_based_auto_wrap_policy(min_num_params=100_000)` wraps each `TransformerBlock` (~50M params) as its own FSDP unit. Embedding/LM head stay in the root unit.
- `ShardingStrategy.FULL_SHARD` = ZeRO-3: params + grads + optimizer states all sharded.
- `MixedPrecision(param_dtype=fp16, reduce_dtype=fp16)`: all-gather/reduce-scatter traffic is fp16, halving inter-node bandwidth usage on Amarel's Ethernet.
- `GradScaler` still required alongside `MixedPrecision` — FSDP's MP policy does not handle fp16 gradient underflow.
- `torch.compile` compiled per-block **before** FSDP wrapping — same ordering rule as Phase 2 DDP.
- Gradient (activation) checkpointing applied **after** FSDP wrapping via `apply_activation_checkpointing()`, reducing peak activation memory by ~√(n_layers) at ~30% extra compute cost.
- Checkpoints use `torch.distributed.checkpoint` (DCP): every rank writes its own shard file, no gathering needed, no OOM.
- Linear LR scaling: `effective_lr = base_lr × world_size = 6e-5 × 16 = 9.6e-4`.
- `RankShardedDataset` shards the TinyStories stream identically to Phase 2 — data parallelism is unchanged.

---

## Training

### Hyperparameters

| Param | Value | Reason |
| :--- | :--- | :--- |
| `batch_size` | 2 per GPU (config ceiling) | `auto_batch_size()` caps at runtime; effective batch = 2 × 16 = 32 |
| `seq_len` | 1024 | Full GPT-2 context window |
| `learning_rate` | 6e-5 (base) | Scaled to 9.6e-4 by linear rule; smaller base than Phase 2 for stability at 1.3B |
| `warmup_steps` | 5000 | Longer ramp for larger model; AdamW moments need more steps to stabilise |
| `max_steps` | 50000 | Sufficient to verify loss < Phase 2 best; full production run would use 200k+ |
| `grad_clip` | 1.0 | Same as Phase 2 |
| `use_compile` | false | Verify RHEL9 GCC before enabling; see `torch_notes.md` §2 |

### Acceptance criteria

- [ ] Script launches via `torchrun` across 4 nodes without errors
- [ ] Overfit test passes on rank 0 — proxy model loss < 0.01
- [ ] Model sharding verified: `mem alloc` per GPU ≈ `total_param_memory / 16` in logs
- [ ] Scaling efficiency > 80% from 1-node (4 GPUs) to 4-node (16 GPUs)  
      `efficiency = (4-node tok/s) / (1-node tok/s × 4)` *(if < 80%, document in `torch_notes.md`)*
- [ ] Gradient checkpointing active (confirmed from log: `"Gradient checkpointing: enabled"`)
- [ ] Sharded checkpoints written via DCP (`run_<ts>/step_*/` contains shard files)
- [ ] Training loss drops below Phase 2 best (**0.72**) within 10k steps

---

## Running on Amarel

### Prerequisites (inherits from Phase 2)

```bash
conda activate llm    # same env — no new packages required
```

### Submit the 4-node FSDP job

```bash
cd /scratch/$USER/amarel-llm-foundry
git pull origin feature/03-fsdp-hpc-sharding
sbatch slurm/train_fsdp_4nodes.sh
```

### Baseline: 1-node FSDP (for scaling efficiency measurement)

Run FSDP on 1 node first to get the single-node throughput baseline before the full 4-node run:

```bash
sbatch --nodes=1 slurm/train_fsdp_4nodes.sh --config configs/phase3_1B.yaml
```

Record the steady-state `tok/s` from the `.out` file, then compute:

```
efficiency = (4-node tok/s) / (1-node tok/s × 4)
```

### Resume from a sharded checkpoint

```bash
sbatch slurm/train_fsdp_4nodes.sh --resume $CHECKPOINT_DIR/run_<timestamp>/step_<N>
```

### Monitor output

```bash
OUT=gpt_fsdp_4nodes_<JOBID>.out
ERR=gpt_fsdp_4nodes_<JOBID>.err

# Did all 4 nodes start cleanly?
grep "Amarel environment ready" $OUT | wc -l          # expect 4

# Loss curve
grep "^step" $OUT

# Memory per rank (sharding health check)
grep "mem alloc" $OUT | head -20

# Any NCCL or OOM issues?
grep -i "error\|nccl\|timeout\|killed\|oom" $OUT $ERR
```

### Copy best checkpoint to durable storage

```bash
# DCP sharded checkpoints — copy the whole directory
cp -r $CHECKPOINT_DIR/run_<ts>/best ~/checkpoints/phase3_best/
```

Phase 4 (`feature/04-eval-closure`) will export this DCP checkpoint to `model.pt` before single-GPU eval and GPT-2 benchmarking. See `instructions/PHASE-04-eval-closure.md`.

---

## Key differences from Phase 2 (DDP)

| Aspect | Phase 2 DDP | Phase 3 FSDP |
| :--- | :--- | :--- |
| Nodes / GPUs | 1 node / 4 GPUs | 4 nodes / 16 GPUs |
| Model params | 354M | 1,313M (~1.3B) |
| Memory per GPU | ~11 GB (full replica) | ~1 GB (1/16 shard) |
| Gradient sync | All-reduce after backward | Reduce-scatter during backward |
| Optimizer states | Full copy per GPU | Sharded (ZeRO-3 equivalent) |
| Mixed precision | `torch.autocast` + `GradScaler` | FSDP `MixedPrecision` + `GradScaler` |
| Checkpointing | `torch.save(model.module.state_dict())` | `torch.distributed.checkpoint.save()` |
| Activation checkpointing | Not needed | Applied per `TransformerBlock` |
| `torch.compile` | Before DDP wrap | Before FSDP wrap (per-block) |
| torchrun rendezvous | `--master_addr=localhost` | c10d rendezvous via `MASTER_ADDR` from SLURM |

---

## Repository structure

```
amarel-llm-foundry/
├── configs/
│   ├── base_config.yaml         # global defaults (shared)
│   ├── phase1_124M.yaml         # Phase 1: 124M model
│   ├── phase2_350M.yaml         # Phase 2: 350M model
│   └── phase3_1B.yaml           # Phase 3: 1.3B model
├── docs/
│   ├── slurm-guide.md
│   ├── torch_notes.md
│   └── maintenance_schedule.md
├── slurm/
│   ├── common.sh
│   ├── train_1gpu.sh            # Phase 1
│   ├── train_ddp_4gpu.sh        # Phase 2
│   ├── train_fsdp_4nodes.sh     # Phase 3
│   ├── generate.sh
│   └── hello_*.sh
└── src/
    ├── tokenizer.py
    ├── data_utils.py            # shared
    ├── model.py                 # shared
    ├── train_single_gpu.py      # Phase 1 (utilities reused by all trainers)
    ├── train_ddp.py             # Phase 2
    ├── train_fsdp.py            # Phase 3
    └── generate.py              # shared
```

---

## Project map

| Phase | Branch | Status | Goal |
| :--- | :--- | :---: | :--- |
| 0 | `feature/00-hpc-distributed-baseline` | ✅ | Validate `torch.distributed` + NCCL across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | ✅ | 124M GPT from scratch, single-GPU, loss 1.03 |
| 2 | `feature/02-ddp-multi-gpu` | ✅ | DDP 4-GPU, 350M model — loss 0.72, coherent text confirmed |
| **3** | `feature/03-fsdp-hpc-sharding` | 🔄 | FSDP 4-node, 1.3B model |
| 4 | `feature/04-eval-closure` | ⬜ | Eval harness, GPT-2 benchmark, project closure |
