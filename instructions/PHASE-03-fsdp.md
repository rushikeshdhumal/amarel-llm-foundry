# PHASE 03: FSDP HPC Sharding (Multi-Node)

**Branch**: `feature/03-fsdp-hpc-sharding`  
**Prerequisite**: `feature/02-ddp-multi-gpu` (working DDP training on single node).  
**Goal**: Scale to multiple Amarel nodes using PyTorch's `FullyShardedDataParallel` (FSDP) to shard model parameters, gradients, and optimizer states across GPUs.

---

## 1. Objective
Replace `DistributedDataParallel` with `FSDP`. Enable model sharding across 4+ nodes, integrate gradient checkpointing and mixed precision, and measure scaling efficiency.

---

## 2. Files to Create / Modify
- `src/train_fsdp.py` – Copy `train_ddp.py`; replace DDP wrapper with FSDP + `size_based_auto_wrap_policy`.
- `src/model.py` – Ensure all modules are FSDP-friendly (wrap `TransformerBlock` individually).
- `configs/phase3_1B.yaml` – New config: `n_layer=32, n_head=20, n_embd=2048` (target ~1B params).
- `slurm/train_fsdp_4nodes.sh` – Multi-node SLURM script: `#SBATCH --nodes=4 --ntasks-per-node=4` (16 GPUs total).

---

## 3. Acceptance Criteria
- [ ] Model sharding verified: `torch.cuda.memory_allocated()` per GPU drops proportionally as nodes increase.
- [ ] Scaling efficiency >80% when going from 1 node to 4 nodes (measured in tokens/second).
- [ ] Gradient checkpointing enabled (`activation_checkpointing_policy`) to fit 1B params on A100/V100.
- [ ] Sharded checkpoints saved via `torch.distributed.checkpoint` (not `model.state_dict()`).
- [ ] Loss converges to the same validation perplexity as Phase 2 (within 0.1).

---

## 4. Amarel Edge Case (Must Handle)
- **Cross-node NCCL**: Set `NCCL_IB_DISABLE=1` and `NCCL_SOCKET_IFNAME=eth0` if InfiniBand is misconfigured.
- **Checkpoint Path**: Save sharded checkpoints to `$SCRATCH/checkpoints/` with `--save-sharded` flag.
- **Resume**: Implement `--resume` to load sharded checkpoints via `load_sharded_state_dict`.
- **Timeout**: Increase `NCCL_TIMEOUT=1800` for large models to prevent premature timeouts on slow interconnect.

---

## 5. Cursor Instructions
- Follow `@GLOBAL.md`: use `torch.cuda.amp` with FSDP's native mixed precision (`MixedPrecisionPolicy`).
- Use `torch.distributed.checkpoint` APIs—never manual `save`/`load` for sharded checkpoints.
- Enable `torch.compile` only after FSDP wrapping (order matters).
- Log per-rank memory usage every 100 steps to debug imbalance. 