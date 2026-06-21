"""
train_ddp.py — Multi-GPU DDP training for GPT on a single Amarel node.

WHAT IS DISTRIBUTEDDATAPARALLEL (DDP)?
───────────────────────────────────────
DDP replicates the full model on every GPU. During training:

  1. Each GPU receives a different mini-batch (data parallelism).
  2. Each GPU independently runs the full forward + backward pass.
  3. After backward, NCCL all-reduces (averages) the gradients across all GPUs.
  4. Every GPU applies identical gradient updates, so all replicas stay in sync.

Why this is fast:
  - Forward and backward are fully parallel across GPUs (no communication).
  - NCCL all-reduce is overlapped with the backward pass: as soon as a bucket
    of gradients is ready, NCCL starts transmitting them while the backward
    continues computing gradients for earlier layers. Communication and
    computation overlap, so the all-reduce is nearly "free".
  - Result: ~N× throughput with N GPUs (minus small all-reduce overhead).

RANK vs LOCAL_RANK vs WORLD_SIZE
─────────────────────────────────
torchrun injects three environment variables into each process:
  RANK        — global rank across ALL nodes (0 … world_size-1)
  LOCAL_RANK  — rank WITHIN the current node (0 … nproc_per_node-1)
  WORLD_SIZE  — total number of GPU processes

On a single node, RANK == LOCAL_RANK. On multi-node (Phase 3), they differ:
  e.g. node 0 has ranks 0,1,2,3 (local 0,1,2,3); node 1 has ranks 4,5,6,7 (local 0,1,2,3).
LOCAL_RANK is the CUDA device index (the physical GPU slot on the node).

LINEAR LR SCALING RULE
───────────────────────
When we increase the effective batch size by N (by using N GPUs), gradients
average over N× more samples per step. To keep the learning signal the same
magnitude, we scale the learning rate by N:

  effective_lr = base_lr × world_size

This is the "linear scaling rule" from Goyal et al. (Facebook, 2017).
It keeps the ratio of lr/batch_size constant, preserving training dynamics.
With world_size=4: effective_lr = 3e-4 × 4 = 1.2e-3.

DATA SHARDING (WHY NOT DistributedSampler)
──────────────────────────────────────────
DistributedSampler requires __len__ and __getitem__ (a map-style dataset).
TinyStories is too large to index in RAM, so we use an IterableDataset
(streaming). RankShardedDataset provides equivalent DDP sharding: rank r
yields sequence i only when i % (world_size × num_workers) lands on rank r's
combined shard, handling both DDP rank splits and DataLoader worker splits
in one pass.

CHECKPOINT SAFETY IN DDP
─────────────────────────
All replicas produce identical gradients → identical weight updates (DDP
guarantees this via all-reduce). But if all ranks wrote checkpoints
simultaneously to the same path, they'd corrupt each other's writes.
Fix: only rank 0 writes; all ranks call dist.barrier() afterwards so no
rank races ahead before the checkpoint is fully flushed to disk.
"""

import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import torch
import torch._dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import OmegaConf, DictConfig

from src.model import GPT, GPTConfig
from src.data_utils import TinyStoriesDataset

# Reuse all pure-utility functions from the single-GPU trainer.
# This follows GLOBAL.md §5: "do not duplicate code; share modules."
from src.train_single_gpu import (
    get_lr,
    save_checkpoint,
    load_checkpoint,
    run_overfit_test,
    build_optimizer,
)


# ── DDP process group lifecycle ────────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize the NCCL process group using environment variables set by torchrun.

    torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    before launching this process. init_process_group(init_method="env://")
    reads them automatically.

    CRITICAL ORDER: torch.cuda.set_device BEFORE init_process_group.
    NCCL needs the device bound before it starts the rendezvous.
    (Same lesson as Phase 0 dist_hello.py fix.)

    Returns:
        (rank, local_rank, world_size)
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Bind this process to its GPU before starting NCCL rendezvous.
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        # device_id enables NCCL's eager connection path — faster startup.
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Tear down the NCCL process group at the end of training."""
    dist.destroy_process_group()


# ── Rank-sharded dataset ───────────────────────────────────────────────────────

class RankShardedDataset(IterableDataset):
    """
    Wraps TinyStoriesDataset to give each DDP rank a non-overlapping shard.

    WHY NOT DistributedSampler?
      DistributedSampler requires __len__ and __getitem__ (map-style dataset).
      TinyStories is streamed via IterableDataset because it is too large to
      index into RAM. This class provides equivalent behavior for streaming.

    HOW the sharding works:
      The base dataset yields a stream of (x, y) pairs, numbered 0, 1, 2, …
      We assign sequence i to the shard with index i % total_shards, where:

        total_shards = world_size × num_dataloader_workers

      Rank r, DataLoader worker w gets sequences where:
        i % total_shards == r × num_workers + w

      This two-level sharding ensures:
        - No two ranks receive the same sequence (data parallelism correctness).
        - Within a rank, no two DataLoader workers fetch duplicate batches.
        - Every sequence is assigned to exactly one (rank, worker) pair.

    WHY create a fresh TinyStoriesDataset per worker?
      Each DataLoader worker runs in its own forked process. Sharing a single
      TinyStoriesDataset instance across workers would share the open file
      handle, causing concurrent reads to corrupt each other. Creating a fresh
      instance per worker gives each worker its own file handle.
    """

    def __init__(
        self,
        file_path: str,
        seq_len: int,
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield (x, y) pairs belonging to this rank's shard.

        Reads get_worker_info() to subdivide this rank's shard further
        across multiple DataLoader workers within the same process.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # No DataLoader worker parallelism (num_workers=0), or the main
            # process is iterating directly.
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # Combined shard index for this (rank, worker) pair.
        total_shards = self.world_size * num_workers
        my_shard = self.rank * num_workers + worker_id

        # Fresh dataset instance → independent file handle per worker.
        base = TinyStoriesDataset(file_path=self.file_path, seq_len=self.seq_len)

        for idx, sample in enumerate(base):
            if idx % total_shards == my_shard:
                yield sample


def build_ddp_dataloader(
    file_path: str,
    seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 2,
) -> DataLoader:
    """
    Build a rank-sharded DataLoader for DDP training.

    drop_last=True: DDP requires every rank to produce the same number of
    batches per step. If one rank has fewer samples (e.g. due to uneven
    sharding), it would call all-reduce fewer times than other ranks → NCCL
    hang. Dropping the last incomplete batch prevents this.

    Args:
        file_path:   Path to TinyStories .txt file.
        seq_len:     Token sequence length per example.
        batch_size:  Per-GPU batch size (not the global effective batch size).
        rank:        This process's global rank.
        world_size:  Total number of DDP processes.
        num_workers: CPU prefetch workers spawned per rank process.

    Returns:
        DataLoader yielding (x, y) of shape (batch_size, seq_len).
    """
    dataset = RankShardedDataset(
        file_path=file_path,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # Essential for DDP: all ranks must produce equal-length batches.
        drop_last=True,
        # pin_memory copies batches into page-locked RAM → faster DMA to GPU.
        pin_memory=True,
        num_workers=num_workers,
        # Keeps worker processes alive between iterations (avoids re-spawn cost).
        persistent_workers=(num_workers > 0),
    )


# ── Main training function ─────────────────────────────────────────────────────

def train(
    cfg: DictConfig,
    rank: int,
    local_rank: int,
    world_size: int,
    resume_dir: Path | None = None,
) -> None:
    """
    DDP training loop.

    Args:
        cfg:         Merged OmegaConf config (base_config + phase2_350M).
        rank:        Global rank of this process.
        local_rank:  Local rank (= GPU index on single-node setups).
        world_size:  Total number of GPU processes.
        resume_dir:  If set, restore weights and optimizer state from this dir.
    """
    device = torch.device(f"cuda:{local_rank}")

    # is_master: rank 0 is the "master" that prints logs and writes checkpoints.
    # All other ranks are silent workers that only train and sync gradients.
    is_master = (rank == 0)

    # ── Determinism ───────────────────────────────────────────────────────────
    # Add rank to the seed so each GPU samples different dropout masks and
    # different DataLoader workers get different random augmentations, while
    # keeping runs reproducible (same seed + rank → same run).
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig.from_omegaconf(cfg)
    model = GPT(model_cfg).to(device)

    if is_master:
        print(f"Model: {model.count_parameters() / 1e6:.1f}M params")
        print(f"Device: {device}  GPU: {torch.cuda.get_device_name(local_rank)}")

    # ── torch.compile BEFORE DDP wrap ─────────────────────────────────────────
    # Compiling the raw module before DDP lets TorchInductor see the model's
    # computation graph in isolation. Compiling after DDP would include DDP's
    # gradient synchronization hooks in the compiled graph, which is harder
    # to optimise and can cause subtle correctness issues.
    use_compile: bool = cfg.get("use_compile", False)
    if use_compile and device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            if is_master:
                print(f"Compiling model (sm_{cap[0]}{cap[1]})...")
            try:
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model)
                if is_master:
                    print("torch.compile: enabled")
            except Exception as e:
                if is_master:
                    print(f"torch.compile failed ({e}), using eager mode")
        else:
            if is_master:
                print(f"torch.compile: skipped (sm_{cap[0]}{cap[1]} < sm_80)")
    else:
        if is_master:
            print(f"torch.compile: disabled (use_compile={use_compile})")

    # ── Wrap with DDP ─────────────────────────────────────────────────────────
    # device_ids=[local_rank]: tells DDP which physical GPU this process owns.
    # find_unused_parameters=False: slightly faster backward because DDP
    # doesn't need to scan for parameters that received no gradient.
    # Only set find_unused_parameters=True if your model has conditional
    # branches that leave some parameters unused in certain forward passes.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer with linear LR scaling ──────────────────────────────────────
    # cfg.learning_rate is the base (single-GPU) learning rate from the config.
    # We scale it by world_size to apply the linear scaling rule.
    # build_optimizer is called on model.module (the raw GPT) because it uses
    # GPT type hints and applies weight-decay grouping by parameter dimensionality.
    scaled_lr = cfg.learning_rate * world_size
    cfg_scaled = OmegaConf.merge(cfg, {"learning_rate": scaled_lr})
    optimizer = build_optimizer(model.module, cfg_scaled)

    if is_master:
        print(
            f"LR: base={cfg.learning_rate:.2e} × world_size={world_size}"
            f" → effective={scaled_lr:.2e}"
        )
        print(
            f"Effective batch size: {cfg.batch_size} × {world_size}"
            f" = {cfg.batch_size * world_size} sequences/step"
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    # ALL ranks load the checkpoint independently. DDP requires all replicas
    # to start from identical weights; loading the same file achieves this.
    start_step = 0
    if resume_dir is not None:
        start_step = load_checkpoint(model.module, optimizer, resume_dir, device)

    # Broadcast start_step from rank 0 to confirm all ranks agree.
    # This catches rare race conditions where one rank fails to load.
    start_step_t = torch.tensor(start_step, device=device)
    dist.broadcast(start_step_t, src=0)
    start_step = int(start_step_t.item())

    # ── Overfit test (rank 0 only) ─────────────────────────────────────────────
    # The test runs a tiny proxy model and is purely local — no communication.
    # Only rank 0 runs it (and prints the result); all ranks barrier afterwards
    # so training starts at the same time.
    if start_step == 0:
        if is_master:
            run_overfit_test(model.module, device)
        dist.barrier()

    # ── Data ──────────────────────────────────────────────────────────────────
    assert os.path.exists(cfg.data_path), (
        f"Training data not found: {cfg.data_path}\n"
        "Run: wget .../TinyStoriesV2-GPT4-train.txt -P $SCRATCH/data/tinystories/"
    )

    train_loader = build_ddp_dataloader(
        file_path=cfg.data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=2,
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    # Each rank manages its own GradScaler independently. DDP's all-reduce
    # averages the fp16-scaled gradients; scaling and unscaling are symmetric
    # across ranks so the effective gradient update is identical to single-GPU.
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ── Checkpoint paths (rank 0 only) ────────────────────────────────────────
    # Only rank 0 writes checkpoints, so only rank 0 needs these paths.
    run_dir: Path | None = None
    best_ckpt_dir: Path | None = None
    best_loss = float("inf")
    if is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_root = Path(
            cfg.get("checkpoint_dir", os.environ.get("CHECKPOINT_DIR", "checkpoints"))
        )
        run_dir = ckpt_root / f"run_{timestamp}"
        best_ckpt_dir = run_dir / "best"

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    data_iter = iter(train_loader)
    step = start_step
    t0 = time.time()

    if is_master:
        print(
            f"\n── Training (steps {start_step} → {cfg.max_steps},"
            f" world_size={world_size}) ──"
        )

    while step < cfg.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            # The sharded stream is exhausted — restart from the beginning.
            # With a large dataset like TinyStories this rarely happens in
            # the first 100k steps.
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Update LR for this step (cosine schedule using the scaled LR).
        lr = get_lr(step, scaled_lr, cfg.warmup_steps, cfg.max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward pass ──────────────────────────────────────────────────────
        # autocast casts eligible ops to fp16; loss stays fp32.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            _, loss = model(x, y)

        # ── Backward pass ─────────────────────────────────────────────────────
        # DDP hooks fire during backward: as each parameter's gradient is
        # computed, it is bucket-queued for all-reduce. Buckets are all-reduced
        # asynchronously while backward continues to earlier layers.
        # By the time backward() returns, all gradients are averaged.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale before clipping so the clip norm threshold is meaningful.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.grad_clip
        )

        scaler.step(optimizer)
        scaler.update()

        step += 1

        # ── Logging (rank 0 only) ──────────────────────────────────────────────
        if is_master and step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            # Report the combined throughput across all GPUs.
            tokens_per_sec = (
                cfg.log_interval * cfg.batch_size * world_size * cfg.seq_len / elapsed
            )
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | tok/s {tokens_per_sec:.0f}"
            )
            t0 = time.time()

        if is_master and step % cfg.grad_norm_log_interval == 0:
            print(f"  grad_norm = {grad_norm:.4f}")

        # ── Checkpointing (rank 0 only) + barrier ─────────────────────────────
        if step % cfg.save_interval == 0:
            if is_master:
                # Save model.module (the raw GPT), not model (the DDP wrapper).
                # Saving the DDP wrapper would add "module." prefixes to every
                # key in the state dict, making the checkpoint incompatible with
                # the plain GPT.load_state_dict() expected by generate.py.
                save_checkpoint(
                    model.module, optimizer, step, loss.item(), cfg,
                    run_dir / f"step_{step:07d}",
                )
            # All ranks pause here until rank 0 finishes writing the checkpoint.
            # Without this barrier, a fast rank could start the next iteration
            # and potentially crash before the write completes.
            dist.barrier()

        # ── Save-best checkpoint (rank 0 only) ────────────────────────────────
        current_loss = loss.item()
        if is_master and current_loss < best_loss:
            best_loss = current_loss
            if best_ckpt_dir.exists():
                shutil.rmtree(best_ckpt_dir)
            save_checkpoint(model.module, optimizer, step, current_loss, cfg, best_ckpt_dir)
        dist.barrier()

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if is_master:
        save_checkpoint(
            model.module, optimizer, step, loss.item(), cfg,
            run_dir / f"step_{step:07d}_final",
        )
        print(f"\nTraining complete. Checkpoints: {run_dir}")

    dist.barrier()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU DDP GPT training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Phase config YAML (e.g. configs/phase2_350M.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint directory to resume from",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override data_path from config",
    )
    return parser.parse_args()


def main() -> None:
    # NCCL must raise errors immediately rather than hanging.
    # TORCH_NCCL_ASYNC_ERROR_HANDLING is the correct name in PyTorch 2.x
    # (the old NCCL_ASYNC_ERROR_HANDLING is silently ignored).
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    rank, local_rank, world_size = setup_distributed()

    args = parse_args()

    base_cfg = OmegaConf.load("configs/base_config.yaml")
    phase_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, phase_cfg)

    if args.data_path is not None:
        cfg.data_path = args.data_path

    if rank == 0:
        print("── Config ──")
        print(OmegaConf.to_yaml(cfg))

    resume_dir = Path(args.resume) if args.resume else None

    try:
        train(cfg, rank, local_rank, world_size, resume_dir)
    finally:
        # Always destroy the process group, even if training raises an exception.
        # Leaving a stale process group causes the other ranks to hang forever.
        cleanup_distributed()


if __name__ == "__main__":
    main()
