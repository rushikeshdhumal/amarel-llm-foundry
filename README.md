# Amarel LLM Foundry

A **5-phase, branch-per-phase** project that builds a GPT-style language model from scratch on [Amarel](https://oarc.rutgers.edu/amarel/) (Rutgers HPC) — from a distributed "hello world" to multi-node FSDP training and, eventually, a multi-agent system.

**Stack:** pure PyTorch (`torch.nn`), SLURM, NCCL, TinyStories, tiktoken. No HuggingFace Trainer, no PyTorch Lightning.

---

## How this repo is organized

Each phase lives on its **own Git branch**. The `main` branch is the **landing page** (this file) plus shared planning docs. All runnable code, phase-specific READMEs, configs, and SLURM scripts live on the feature branches.

```
main                          ← you are here (overview only)
 │
 ├── feature/00-hpc-distributed-baseline
 ├── feature/01-nanogpt-transformer
 ├── feature/02-ddp-multi-gpu
 ├── feature/03-fsdp-hpc-sharding
 └── feature/04-agentic-system
```

**Workflow for each phase:**

1. Check out the phase branch.
2. Read that branch's `README.md` (acceptance criteria, run commands, measured results).
3. Follow `instructions/PHASE-XX-*.md` for deliverables and Cursor agent guidance.
4. Merge or rebase the previous phase branch when starting the next one (each phase lists its prerequisite).

```bash
git clone https://github.com/rushikeshdhumal/amarel-llm-foundry.git
cd amarel-llm-foundry

# Example: start Phase 0
git checkout feature/00-hpc-distributed-baseline
cat README.md
cat instructions/PHASE-00-hpc-baseline.md
```

Phases are **sequential** — later phases reuse modules from earlier ones (`model.py`, `data_utils.py`, training utilities) rather than reimplementing them.

---

## Project map

| Phase | Branch | What you build | Scale |
| :---: | :--- | :--- | :--- |
| **0** | [`feature/00-hpc-distributed-baseline`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/00-hpc-distributed-baseline) | SLURM + `torch.distributed` smoke tests across Amarel nodes | NCCL hello-world |
| **1** | [`feature/01-nanogpt-transformer`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/01-nanogpt-transformer) | GPT-2 transformer from scratch, single-GPU training, inference | ~124M params, 1 GPU |
| **2** | [`feature/02-ddp-multi-gpu`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/02-ddp-multi-gpu) | `DistributedDataParallel` on one node | ~350M params, 4 GPUs |
| **3** | [`feature/03-fsdp-hpc-sharding`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/03-fsdp-hpc-sharding) | `FullyShardedDataParallel` across nodes, sharded checkpoints | ~1.3B params, 4 nodes × 4 GPUs |
| **4** | [`feature/04-agentic-system`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/04-agentic-system) | ReAct-style agents using the trained checkpoint | inference + tools |

Instruction blueprints for every phase sit in [`instructions/`](instructions/). Shared agent rules: [`instructions/GLOBAL.md`](instructions/GLOBAL.md).

---

## Architecture at a glance

Every training phase uses the same GPT-2-style decoder-only stack:

```
Token IDs → Embedding + Positional encoding
         → × L  TransformerBlock (LayerNorm → Attention → MLP, with residuals)
         → LayerNorm → LM head → next-token logits
```

| Phase | Model | Parallelism | Key idea |
| :---: | :--- | :--- | :--- |
| 1 | 124M | 1 GPU | Baseline training loop, AMP, checkpoints |
| 2 | 350M | DDP (full replica per GPU) | Data parallelism + linear LR scaling |
| 3 | 1.3B | FSDP (sharded params/grads/optimizer) | Model too large to replicate; shard across 16 GPUs |

---

## Getting started on Amarel

**Prerequisites:** Amarel account, conda env with PyTorch + CUDA, dataset on scratch:

```bash
# On Amarel login node — one-time setup
module use /projects/community/modulefiles
module load cuda/12.8.1
conda create -n llm python=3.11 -y && conda activate llm
pip install -e .

# Dataset (used from Phase 1 onward)
export SCRATCH=/scratch/$USER
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt \
     -P $SCRATCH/data/tinystories/
```

**Typical job flow** (details vary by phase — see branch README):

```bash
cd /scratch/$USER/amarel-llm-foundry
git fetch && git checkout feature/01-nanogpt-transformer   # pick your phase

sbatch slurm/train_1gpu.sh          # Phase 1 example
sbatch slurm/train_ddp_4gpu.sh      # Phase 2
sbatch slurm/train_fsdp_4nodes.sh   # Phase 3
```

Checkpoints and data go on **`$SCRATCH`** (not `$HOME`) — scratch is fast but not backed up; copy best checkpoints to `~/checkpoints/` for durability.

---

## What's on `main` vs feature branches

| Path | On `main` | On feature branches |
| :--- | :---: | :---: |
| `README.md` | Overview (this file) | Phase-specific progress & runbook |
| `instructions/` | Phase blueprints | Same |
| `docs/` | Shared guides | Same (+ phase notes as added) |
| `src/` | — | Training, model, data code |
| `configs/` | — | YAML hyperparameters per phase |
| `slurm/` | — | Batch scripts + `common.sh` |

---

## Documentation

| Doc | Purpose |
| :--- | :--- |
| [`docs/slurm-guide.md`](docs/slurm-guide.md) | SLURM, `torchrun`, multi-node rendezvous, Amarel partitions |
| [`docs/torch_notes.md`](docs/torch_notes.md) | PyTorch gotchas (OOM, NCCL, FSDP, compile, imports) |
| [`docs/maintenance_schedule.md`](docs/maintenance_schedule.md) | Amarel maintenance windows & RHEL9 migration notes |
| [`instructions/GLOBAL.md`](instructions/GLOBAL.md) | Rules for all phases (paths, checkpointing, debugging) |

---

## License

MIT — see [`LICENSE`](LICENSE).
