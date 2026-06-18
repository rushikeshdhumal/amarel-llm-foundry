# PHASE 02: DDP Multi-GPU (Single Node)

**Branch**: `feature/02-ddp-multi-gpu`  
**Prerequisite**: `feature/01-nanogpt-transformer` (must have working `model.py` and `train_single_gpu.py`).  
**Goal**: Scale training to multiple GPUs on a single Amarel node using PyTorch `DistributedDataParallel` (DDP).

---

## 1. Objective
Wrap the existing model with `DistributedDataParallel`. Launch via `torchrun` across 4 GPUs. Verify that loss curves match the single-GPU baseline and that memory is distributed across devices.

---

## 2. Files to Create / Modify
- `src/train_ddp.py` – Copy `train_single_gpu.py`; add DDP wrapper, `DistributedSampler`, and `torchrun`-aware rank initialization.
- `configs/phase2_350M.yaml` – New config: `n_layer=24, n_head=16, n_embd=1024` (scales to 350M params).
- `slurm/train_ddp_4gpu.sh` – Single-node SLURM script with `#SBATCH --nodes=1 --ntasks-per-node=4` using `torchrun`.

---

## 3. Acceptance Criteria
- [ ] Script launches via `torchrun --nproc_per_node=4` (no manual `mpirun`).
- [ ] `DistributedSampler` shuffles data identically across ranks (set seed per epoch).
- [ ] Loss values are identical (within 1e-4) to the single-GPU Phase 1 baseline after 100 steps.
- [ ] Training on 4 GPUs achieves ~3.5x speedup over 1 GPU (accounting for communication overhead).
- [ ] Checkpoints are saved only from rank 0 to avoid file corruption.

---

## 4. Amarel Edge Case (Must Handle)
- `torchrun` auto-sets `MASTER_ADDR` and `MASTER_PORT` for single-node; no manual configuration needed.
- Scale learning rate linearly: `lr = base_lr * world_size` (if using AdamW).
- Ensure `DistributedSampler` uses `shuffle=True` and `drop_last=True` to keep batch sizes even.

---

## 5. Cursor Instructions
- Follow `@GLOBAL.md`: use `torch.cuda.amp`, `torch.compile` on each rank.
- Refactor `train_single_gpu.py` to `train_ddp.py`—do not duplicate code; share `model`, `data`, and `config` modules.
- Use `args.rank` to conditionally print logs and save checkpoints (rank 0 only).
- Test with `torch.distributed.barrier()` to sync checkpoints and validation. 