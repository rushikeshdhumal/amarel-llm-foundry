# Phase 2 — DDP Multi-GPU Training

**Branch**: `feature/02-ddp-multi-gpu`  
**Project**: [Amarel LLM Foundry](https://github.com/rushikeshdhumal/amarel-llm-foundry) — 5-phase journey from SLURM hello-world to a multi-agent system.

---

## Goal

Scale the Phase 1 GPT training to 4 GPUs on a single Amarel node using PyTorch `DistributedDataParallel` (DDP). Train a larger 350M-parameter model and verify that throughput scales close to 4×.

---

## What was built

| File | Purpose |
| :--- | :--- |
| `src/train_ddp.py` | DDP training loop: NCCL process group setup, `RankShardedDataset`, linear LR scaling, rank-0-only checkpointing |
| `src/generate.py` | Inference script (carried from Phase 1 — checkpoints are compatible) |
| `configs/phase2_350M.yaml` | Phase overrides: `n_layer=24`, `n_head=16`, `n_embd=1024` (354M params) |
| `slurm/train_ddp_4gpu.sh` | Single-node 4-GPU SLURM job via `torchrun` |

All model, data, and config utilities (`src/model.py`, `src/data_utils.py`, `src/train_single_gpu.py`) are shared from Phase 1 — no duplication.

---

## Model architecture (350M)

```
Token embedding  (50257 × 1024)
Position embedding (1024 × 1024)
        ↓
 × 24  TransformerBlock
        ├── LayerNorm
        ├── CausalSelfAttention  (16 heads, head_dim=64, causal mask)
        ├── LayerNorm
        └── MLP  (1024 → 4096 → 1024, GELU)
        ↓
LayerNorm → Linear (1024 → 50257, weight-tied to token embedding)
```

Total parameters: **354M** (~350M)

---

## How DDP training works

```
┌─────────────────────────────────────────────────────────────┐
│                      Single SLURM node                      │
│                                                             │
│  torchrun spawns 4 processes (one per GPU):                 │
│                                                             │
│  Rank 0 (GPU 0)   Rank 1 (GPU 1)   Rank 2 (GPU 2)   Rank 3 │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────┐│
│  │  GPT 350M│     │  GPT 350M│     │  GPT 350M│     │ GPT ││
│  │  copy 0  │     │  copy 1  │     │  copy 2  │     │copy3││
│  └────┬─────┘     └────┬─────┘     └────┬─────┘     └──┬──┘│
│       │ batch A        │ batch B        │ batch C       │D  │
│       ↓                ↓                ↓               ↓   │
│   forward+backward  forward+backward  forward+backward  f+b │
│       │ grads          │ grads          │ grads         │   │
│       └────────────────┴────────────────┴───────────────┘   │
│                    NCCL all-reduce (avg grads)               │
│       ┌────────────────┬────────────────┬───────────────┐   │
│       ↓                ↓                ↓               ↓   │
│   optimizer step    optimizer step   optimizer step   opt   │
│   (identical on all 4 GPUs — replicas stay in sync)         │
└─────────────────────────────────────────────────────────────┘
```

**Key design points:**
- `RankShardedDataset`: each rank receives sequence `i` only when `i % world_size == rank`. Non-overlapping by construction; no `DistributedSampler` needed (IterableDataset).
- Linear LR scaling: `effective_lr = base_lr × world_size = 3e-4 × 4 = 1.2e-3`.
- `torch.compile` fires **before** `DDP()` wrap so TorchInductor sees the raw computation graph.
- Checkpoints save `model.module.state_dict()` — strips the DDP `module.` prefix so checkpoints load into plain `GPT` (compatible with `generate.py`).
- Only rank 0 writes; `dist.barrier()` after every save so no rank races ahead.

---

## Training

### Hyperparameters

| Param | Value | Reason |
| :--- | :--- | :--- |
| `batch_size` | 4 per GPU (config ceiling) | `auto_batch_size()` caps this at runtime based on actual GPU VRAM; see `torch_notes.md` §10 |
| `seq_len` | 1024 | Full GPT-2 context window |
| `learning_rate` | 3e-4 (base) | Scaled to 1.2e-3 by `train_ddp.py` (linear scaling rule) |
| `grad_clip` | 1.0 | `clip_grad_norm_` before every optimizer step |
| `warmup_steps` | 2000 | Linear ramp from 0 to effective_lr; matches Phase 1 for comparison |
| `use_compile` | false | Amarel GCC 4.8.5 incompatible with TorchInductor |
| `use_amp` | true (auto) | `torch.autocast(fp16)` + `GradScaler` on all ranks |

### Baseline validation log (124M model, 4-GPU DDP)

Ran `train_ddp_4gpu.sh --config configs/phase1_124M.yaml` first to validate DDP
correctness before committing to the full 350M run.

| Metric | Phase 1 (1 GPU, bs=16) | Phase 2 baseline (4 GPU, bs=16/GPU, 124M) | Notes |
| :--- | :--- | :--- | :--- |
| Best loss | 1.0330 @ step 91,900 | **0.8216** @ step 99,400 | Better: effective bs=64, scaled LR |
| Throughput | ~46,274 tok/s | ~165,074 tok/s | Combined across 4 GPUs |
| Speedup | — | **3.57×** | Target was ≥ 3.5× ✅ |
| Scaling efficiency | — | 89% | 3.57/4 = 0.893 |

Loss is lower than Phase 1 (not identical) because the effective batch size is 4×
larger (64 vs 16) with a linearly scaled LR — more stable gradients per step lead
to better convergence. The "identical within 1e-4" criterion applies only when keeping
the same effective batch; with the linear scaling rule, divergence from step 1 is expected and correct.

### Acceptance criteria

- [x] Script launches via `torchrun --nproc_per_node=4` without errors
- [x] Overfit test passes on rank 0 — proxy model final loss **0.0034** (< 0.01 threshold)
- [x] DDP converges below Phase 1 baseline — best loss **0.8216** vs 1.0330
- [x] Throughput: **3.57×** speedup over single-GPU (165k vs 46k tok/s), ≥ 3.5× target met
- [x] Checkpoints saved only from rank 0 (`model.module.state_dict()`, no `module.` prefix)
- [ ] `generate.py` loads Phase 2 checkpoint and produces coherent text (pending 350M run)

---

## Running on Amarel

### Prerequisites (inherits from Phase 1)

```bash
# Same conda env as Phase 1 — no new packages required
conda activate llm
```

### Submit a training job

```bash
cd /scratch/$USER/amarel-llm-foundry
git pull origin feature/02-ddp-multi-gpu
sbatch slurm/train_ddp_4gpu.sh
```

### Baseline comparison (DDP with 124M config)

To verify DDP reproduces the single-GPU loss curve before committing to the full 350M run:

```bash
sbatch slurm/train_ddp_4gpu.sh --config configs/phase1_124M.yaml
```

Run 100 steps and compare loss to the Phase 1 log. Expect them to match within 1e-4 at each step (same architecture, same effective batch, linear-scaled LR).

### Resume from a checkpoint

```bash
sbatch slurm/train_ddp_4gpu.sh --resume $CHECKPOINT_DIR/run_<timestamp>/step_<N>
```

### Monitor output

```bash
OUT=gpt_ddp_4gpu_<JOBID>.out
tail -50 $OUT                                                           # did it start cleanly?
grep "^step" $OUT | awk -F'|' '{print $2,$0}' | sort -n | head -5 | cut -d' ' -f3-  # best losses
grep -i "error\|nccl\|timeout\|killed\|oom" $OUT gpt_ddp_4gpu_<JOBID>.err
```

### Copy best checkpoint to durable storage

```bash
# Phase 2+ runs produce a best/ directory automatically
cp -r $CHECKPOINT_DIR/run_<ts>/best ~/checkpoints/phase2_best/
```

---

## Repository structure

```
amarel-llm-foundry/
├── configs/
│   ├── base_config.yaml       # global defaults (shared)
│   ├── phase1_124M.yaml       # Phase 1 overrides (shared for baseline check)
│   └── phase2_350M.yaml       # Phase 2: 350M model
├── docs/
│   ├── slurm-guide.md
│   └── torch_notes.md
├── slurm/
│   ├── common.sh
│   ├── train_1gpu.sh          # Phase 1 (single GPU)
│   ├── train_ddp_4gpu.sh      # Phase 2 (4-GPU DDP)
│   ├── generate.sh
│   └── hello_*.sh
└── src/
    ├── tokenizer.py
    ├── data_utils.py          # shared
    ├── model.py               # shared
    ├── train_single_gpu.py    # Phase 1 (utility functions reused by train_ddp.py)
    ├── train_ddp.py           # Phase 2
    └── generate.py            # shared
```

---

## Project map

| Phase | Branch | Status | Goal |
| :--- | :--- | :---: | :--- |
| 0 | `feature/00-hpc-distributed-baseline` | ✅ | Validate `torch.distributed` + NCCL across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | ✅ | 124M GPT from scratch, single-GPU, loss 1.03 |
| **2** | `feature/02-ddp-multi-gpu` | 🔄 | DDP scaling to 4 GPUs, 350M model |
| 3 | `feature/03-fsdp-hpc-sharding` | ⬜ | FSDP multi-node sharding (1B params) |
| 4 | `feature/04-agentic-system` | ⬜ | ReAct agents with trained checkpoint |
