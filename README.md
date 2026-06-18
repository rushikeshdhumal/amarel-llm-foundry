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

## Phase 0 — HPC Distributed Baseline `(current)`

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
- [ ] `hello_1node_4gpu.sh` — ranks 0–3 all on the same hostname
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
