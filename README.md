# Phase 0 — HPC Distributed Baseline

**Branch**: `feature/00-hpc-distributed-baseline`  
**Project**: [Amarel LLM Foundry](https://github.com/rushikeshdhumal/amarel-llm-foundry) — 5-phase journey from SLURM hello-world to scaled GPT training and reproducible evaluation on Amarel HPC.

---

## Goal

Confirm that `torch.distributed` + NCCL are correctly wired across Amarel GPU nodes before any model training begins. A failed distributed setup wastes GPU hours silently — this phase catches those failures early with a minimal smoke test.

---

## What was built

| File | Purpose |
| :--- | :--- |
| `src/dist_hello.py` | Distributed hello-world: each rank prints its world size, hostname, local rank, and GPU name; a cross-rank `dist.barrier()` verifies all-reduce connectivity |
| `slurm/common.sh` | Shared environment: loads CUDA 12.8.1, activates `llm` conda env, exports NCCL variables |
| `slurm/hello_1node_1gpu.sh` | 1 node × 1 GPU smoke test |
| `slurm/hello_1node_4gpu.sh` | 1 node × 4 GPU test |
| `slurm/hello_2nodes_4gpu.sh` | 2 nodes × 2 GPU/node cross-node NCCL test |
| `docs/slurm-guide.md` | SLURM reference guide for Amarel |

---

## Acceptance criteria

- [x] `hello_1node_1gpu.sh` — `[Rank 0/1] Host: gpu030  device=NVIDIA L40S`
- [x] `hello_1node_4gpu.sh` — Ranks 0–3 all on `gpuk002` (A100-PCIE-40GB)
- [x] `hello_2nodes_4gpu.sh` — Ranks 0–3 distributed across `gpuk002` + `gpuk003`; `All ranks healthy.`

### Verified outputs

**`hello_1node_1gpu.sh`**
```
[Rank 0/1] Host: gpu030.amarel.rutgers.edu  local_rank=0  device=NVIDIA L40S
All ranks healthy. Destroying process group.
```

**`hello_1node_4gpu.sh`**
```
[Rank 0/4] Host: gpuk002  local_rank=0  device=NVIDIA A100-PCIE-40GB
[Rank 1/4] Host: gpuk002  local_rank=1  device=NVIDIA A100-PCIE-40GB
[Rank 2/4] Host: gpuk002  local_rank=2  device=NVIDIA A100-PCIE-40GB
[Rank 3/4] Host: gpuk002  local_rank=3  device=NVIDIA A100-PCIE-40GB
All ranks healthy. Destroying process group.
```

**`hello_2nodes_4gpu.sh`**
```
MASTER_ADDR=192.168.19.2  MASTER_PORT=29500
SLURM_NNODES=2  SLURM_NTASKS=2
NCCL version 2.21.5+cuda12.4
[Rank 0/4] Host: gpuk002  local_rank=0  device=NVIDIA A100-PCIE-40GB
[Rank 1/4] Host: gpuk002  local_rank=1  device=NVIDIA A100-PCIE-40GB
[Rank 2/4] Host: gpuk003  local_rank=0  device=NVIDIA A100-PCIE-40GB
[Rank 3/4] Host: gpuk003  local_rank=1  device=NVIDIA A100-PCIE-40GB
All ranks healthy. Destroying process group.
```

---

## NCCL configuration (Amarel-specific)

Getting multi-node NCCL to work required three non-obvious fixes. All are applied in `slurm/common.sh` and `src/dist_hello.py`.

| Fix | Why it was needed |
| :--- | :--- |
| `NCCL_SOCKET_IFNAME=^lo,^docker` | Amarel nodes use `ib0` (InfiniBand) or `eth`-style names that differ by node. Hardcoding `eth0` caused `ncclInvalidUsage`. Excluding loopback/docker lets NCCL auto-pick the right interface. |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | The old name `NCCL_ASYNC_ERROR_HANDLING` is ignored in PyTorch 2.x. Without this, NCCL errors hang silently instead of raising immediately. |
| `NCCL_DEBUG=WARN` | Suppresses `INFO`-level noise while still surfacing real errors in job logs. |
| `torch.cuda.set_device(local_rank)` **before** `init_process_group` | NCCL needs the device set before the rendezvous. Setting it after caused `ncclInvalidUsage` on the barrier. |
| `device_id=torch.device(f"cuda:{local_rank}")` passed to `init_process_group` | Enables NCCL's eager connection path (faster startup, required for correct device association in PyTorch 2.x). |
| `--ntasks-per-node=1` in `hello_2nodes_4gpu.sh` | `srun` with `ntasks-per-node=2` launches two independent `torchrun` processes on the same node — they fight over port 29500 (`EADDRINUSE`). One `torchrun` per node is correct; `torchrun` spawns its own workers internally via `--nproc_per_node`. |

---

## Running the jobs

```bash
cd /scratch/$USER/amarel-llm-foundry
git pull origin feature/00-hpc-distributed-baseline

sbatch slurm/hello_1node_1gpu.sh
sbatch slurm/hello_1node_4gpu.sh
sbatch slurm/hello_2nodes_4gpu.sh

# Monitor queue
squeue -u $USER

# Check output (replace JOBID)
tail -20 hello_1n1g_<JOBID>.out
grep -i "error\|nccl\|timeout\|killed" hello_2n4g_<JOBID>.out hello_2n4g_<JOBID>.err
```

See `docs/slurm-guide.md` for full SLURM reference.

---

## Environment

| Item | Value |
| :--- | :--- |
| Cluster | Amarel (OARC, Rutgers University) |
| GPU | NVIDIA L40S / A100-PCIE-40GB |
| CUDA | 12.8.1 |
| NCCL | 2.21.5+cuda12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 (conda env `llm`) |
| Conda module | `anaconda/2025.06-ts840` |

### First-time cluster setup

```bash
module use /projects/community/modulefiles
module load anaconda/2025.06-ts840
conda create -n llm python=3.11 -y
conda activate llm
conda install numpy -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Repository structure

```
amarel-llm-foundry/
├── docs/
│   └── slurm-guide.md        # SLURM reference for Amarel
├── slurm/
│   ├── common.sh             # shared env (CUDA, conda, NCCL variables)
│   ├── hello_1node_1gpu.sh
│   ├── hello_1node_4gpu.sh
│   └── hello_2nodes_4gpu.sh
└── src/
    └── dist_hello.py         # distributed hello-world script
```

---

## Project map

| Phase | Branch | Status | Goal |
| :--- | :--- | :---: | :--- |
| **0** | `feature/00-hpc-distributed-baseline` | ✅ | Validate `torch.distributed` + NCCL across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | ⬜ | 124M GPT from scratch, single-GPU |
| 2 | `feature/02-ddp-multi-gpu` | ⬜ | DDP 4-GPU, 350M model |
| 3 | `feature/03-fsdp-hpc-sharding` | ⬜ | FSDP 4-node, 1.3B model |
| 4 | `feature/04-eval-closure` | ⬜ | Eval harness, GPT-2 benchmark, project closure |
