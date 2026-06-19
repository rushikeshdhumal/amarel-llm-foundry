# Amarel LLM Foundry

**5-phase journey**: From SLURM hello-world → pure PyTorch transformer → DDP → FSDP multi-node → multi-agent system.

## Project Map

| Phase | Branch | Status | Goal |
| :--- | :--- | :---: | :--- |
| **0** | `feature/00-hpc-distributed-baseline` | ✅ | Validate `torch.distributed` across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | ⬜ | 124M GPT from scratch, single-GPU |
| 2 | `feature/02-ddp-multi-gpu` | ⬜ | DDP scaling to 4 GPUs |
| 3 | `feature/03-fsdp-hpc-sharding` | ⬜ | FSDP multi-node sharding (1B params) |
| 4 | `feature/04-agentic-system` | ⬜ | ReAct agents with trained checkpoint |

---

## Phase 1 — NanoGPT Transformer `(current)`

**Branch**: `feature/01-nanogpt-transformer`

**Goal**: Implement a GPT-2 124M decoder-only transformer from scratch and train it on TinyStories on a single GPU.

### What was built

| File | Purpose |
| :--- | :--- |
| `src/tokenizer.py` | tiktoken GPT-2 BPE wrapper |
| `src/data_utils.py` | Streaming `IterableDataset` over TinyStories `.txt` files |
| `src/model.py` | Full GPT: `CausalSelfAttention` → `MLP` → `TransformerBlock` → `GPT` (124.4M params) |
| `src/train_single_gpu.py` | Training loop: AMP, cosine LR schedule, grad clipping, checkpointing, `--resume` |
| `configs/phase1_124M.yaml` | Phase hyperparameters: `n_layer=12`, `n_head=12`, `n_embd=768`, `bs=16` |
| `slurm/train_1gpu.sh` | Single-GPU SLURM job (16h wall clock) |
| `docs/torch_notes.md` | PyTorch gotchas encountered during development |

### Acceptance criteria

- [x] `model.py` forward pass runs without shape errors for a dummy batch (`bs=4, seq=1024`)
- [ ] Single-batch overfit test: loss drops to < 0.01 within 100 steps
- [ ] Full training on TinyStories for 1 hour produces coherent text completions
- [x] Checkpoints saved to `checkpoints/run_{timestamp}/` with `model.pt`, `config.yaml`, `optimizer.pt`
- [x] Training script accepts `--config` and `--resume` flags
- [ ] Gradient norms logged every `grad_norm_log_interval` steps

### Training log

| Run | Steps | Best loss | Notes |
| :--- | :--- | :--- | :--- |
| Run 1 | 0 → 27,500 | **1.2557** (step 26,600) | Killed by 4h wall-clock limit; resumed from step 27,500 |

### Environment

| Item | Value |
| :--- | :--- |
| Cluster | Amarel (OARC, Rutgers University) |
| GPU | NVIDIA L40S / A100-PCIE-40GB (40 GB) |
| CUDA | 12.8.1 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 (conda env `llm`) |

### Resuming training

```bash
cd /scratch/$USER/amarel-llm-foundry
git pull origin feature/01-nanogpt-transformer
sbatch slurm/train_1gpu.sh --resume $CHECKPOINT_DIR/run_<timestamp>/step_<N>
```

---

## Phase 0 — HPC Distributed Baseline

**Branch**: `feature/00-hpc-distributed-baseline`

**Goal**: Confirm that `torch.distributed` + NCCL are correctly wired across Amarel GPU nodes before any model training begins.

### What was built

| File | Purpose |
| :--- | :--- |
| `src/dist_hello.py` | Distributed hello-world: prints rank, world size, hostname, and GPU name across all processes |
| `slurm/hello_1node_1gpu.sh` | 1 node × 1 GPU smoke test |
| `slurm/hello_1node_4gpu.sh` | 1 node × 4 GPU test |
| `slurm/hello_2nodes_4gpu.sh` | 2 nodes × 2 GPU cross-node NCCL test |
| `slurm/common.sh` | Shared environment: loads CUDA 12.8.1, activates `llm` conda env, exports NCCL variables |
| `docs/slurm-guide.md` | SLURM reference guide for this cluster |

### Acceptance criteria

- [x] `hello_1node_1gpu.sh` — `[Rank 0/1] Host: gpu030.amarel.rutgers.edu  device=NVIDIA L40S`
- [x] `hello_1node_4gpu.sh` — ranks 0–3 all on `gpuk002.amarel.rutgers.edu` (NVIDIA A100-PCIE-40GB)
- [ ] `hello_2nodes_4gpu.sh` — ranks distributed across 2 distinct hostnames

### Environment

| Item | Value |
| :--- | :--- |
| Cluster | Amarel (OARC, Rutgers University) |
| GPU | NVIDIA L40S |
| CUDA | 12.8.1 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 (conda env `llm`) |
| Conda module | `anaconda/2025.06-ts840` |

### Cluster setup (first-time only)

```bash
# On the login node
module use /projects/community/modulefiles
module load anaconda/2025.06-ts840
conda create -n llm python=3.11 -y
conda activate llm
conda install numpy -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Running the jobs

```bash
cd /scratch/$USER/amarel-llm-foundry

sbatch slurm/hello_1node_1gpu.sh
sbatch slurm/hello_1node_4gpu.sh
sbatch slurm/hello_2nodes_4gpu.sh

# Monitor
squeue -u $USER
tail -f hello_1n1g_<JOBID>.out   # Ctrl+C to exit tail
```

See `docs/slurm-guide.md` for full SLURM reference.

---

## Repository Structure

```
amarel-llm-foundry/
├── configs/          # YAML hyperparameter configs (shared across phases)
├── docs/             # Guides and notes (slurm-guide.md, ...)
├── instructions/     # Cursor agent blueprints for each phase
├── slurm/            # SLURM job scripts + common.sh
└── src/              # Python source (grows each phase)
```
