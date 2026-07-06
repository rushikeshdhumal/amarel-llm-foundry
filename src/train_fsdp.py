"""
train_fsdp.py — Multi-node FSDP training for GPT on Amarel.

WHY FSDP INSTEAD OF DDP?
─────────────────────────
DDP replicates the full model on every GPU. Each rank holds:
  - All parameters      (fp32): n_params × 4 B
  - All gradients       (fp32): n_params × 4 B
  - AdamW first moment  (fp32): n_params × 4 B
  - AdamW second moment (fp32): n_params × 4 B
  Total: n_params × 16 B

For a 1.3B model: 1.3B × 16 B ≈ 20 GB per GPU — impossible on a 40 GB A100
once activations and NCCL buffers are included.

FSDP (FullyShardedDataParallel) solves this by sharding all four across every
rank. With 16 GPUs, each rank holds only 1/16th of each tensor:
  - Parameter shard    (fp16): 1.3B × 2 B / 16 ≈ 163 MB
  - Optimizer shard    (fp32): 1.3B × 8 B / 16 ≈ 653 MB
  Total per rank: < 1 GB — fits easily on 40 GB A100.

HOW FSDP WORKS
───────────────
FSDP wraps designated module boundaries (here: each TransformerBlock) and
manages three collective operations transparently:

  FORWARD PASS:
    For each FSDP-wrapped block (in order):
      1. All-gather: every rank broadcasts its parameter shard to all others,
         reconstructing the full block weights temporarily.
      2. Run the forward pass on the full (local) weights.
      3. Discard the gathered weights — only the shard is kept.

  BACKWARD PASS:
    For each block (in reverse order):
      1. All-gather: re-gather the full weights (needed to compute gradients).
      2. Run the backward pass.
      3. Reduce-scatter: each rank accumulates only its own gradient shard.
      4. Discard the gathered weights again.

  OPTIMIZER STEP:
    Each rank runs AdamW only on its own parameter and gradient shards.
    No communication needed — all ranks independently update their slice.

COMMUNICATION vs DDP:
  DDP: one all-reduce of size model_size per backward pass.
  FSDP: n_blocks all-gathers (forward) + n_blocks reduce-scatters (backward),
  each of size block_params. Total data moved is the same (≈ model_size),
  but FSDP spreads it across many smaller collectives, which pipeline better
  with compute over slow Ethernet.

MIXED PRECISION IN FSDP
─────────────────────────
We use FSDP's native MixedPrecision policy instead of torch.autocast:
  - MixedPrecision casts parameters to fp16 DURING the all-gather, so the
    communication bandwidth is halved (fp16 vs fp32 weights).
  - torch.autocast would not know about FSDP's gather/scatter, leaving those
    in fp32 and wasting bandwidth.
  - GradScaler is still required because fp16 gradients can underflow —
    FSDP's MP policy does not handle loss scaling.

GRADIENT CHECKPOINTING
───────────────────────
Gradient checkpointing (activation checkpointing) trades compute for memory:
instead of storing all intermediate activations for the backward pass, it
recomputes them on demand. For each TransformerBlock, FSDP's all-gather already
discards weights after forward — activation checkpointing additionally discards
activations, reducing peak memory by ~√(n_layers) at the cost of one extra
forward pass per block.

Applied AFTER FSDP wrapping, using apply_activation_checkpointing() which
wraps each TransformerBlock's forward with torch.utils.checkpoint.checkpoint().

CHECKPOINT SAFETY WITH FSDP
─────────────────────────────
FSDP shards parameters across GPUs. Using torch.save(model.state_dict()) with
the default (FULL) state dict type forces rank 0 to gather all 1.3B params in
fp32 (20 GB) — OOM. Instead, we use torch.distributed.checkpoint (DCP) which
writes one shard file per rank directly to disk, never gathering the full model.
"""

import argparse
import gc
import os
import shutil
import time
import types
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Iterator

import torch
import torch._dynamo
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import OmegaConf, DictConfig
import yaml

from src.model import GPT, GPTConfig, TransformerBlock
from src.data_utils import TinyStoriesDataset

# Reuse pure-utility functions from the single-GPU trainer.
# Note: load_checkpoint is intentionally NOT imported — FSDP uses
# load_fsdp_checkpoint (DCP-based) instead of the single-GPU loader.
from src.train_single_gpu import (
    get_lr,
    run_overfit_test,
    build_optimizer,
)


# ── DDP/FSDP process group lifecycle ──────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int]:
    """
    Initialise the NCCL process group using environment variables set by torchrun.

    For multi-node jobs, torchrun sets MASTER_ADDR to the hostname of node 0.
    All nodes rendezvous at MASTER_ADDR:MASTER_PORT before any collective runs.

    Critical order (same as Phase 2):
      1. torch.cuda.set_device(local_rank)  — bind GPU before NCCL communicator
      2. dist.init_process_group(...)       — build communicator with device known

    Returns:
        (rank, local_rank, world_size)
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Tear down the NCCL process group."""
    dist.destroy_process_group()


# ── Rank-sharded dataset (identical to Phase 2) ───────────────────────────────

class RankShardedDataset(IterableDataset):
    """
    Wraps TinyStoriesDataset so each DDP/FSDP rank receives a non-overlapping shard.
    Identical logic to Phase 2 — sharding is a data property, not a model property.
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
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        total_shards = self.world_size * num_workers
        my_shard = self.rank * num_workers + worker_id

        base = TinyStoriesDataset(file_path=self.file_path, seq_len=self.seq_len)
        for idx, sample in enumerate(base):
            if idx % total_shards == my_shard:
                yield sample


def build_fsdp_dataloader(
    file_path: str,
    seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 2,
) -> DataLoader:
    dataset = RankShardedDataset(
        file_path=file_path,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


# ── GPU memory-aware batch size ────────────────────────────────────────────────

def auto_batch_size(
    requested: int,
    model_cfg: GPTConfig,
    n_params: int,
    world_size: int,
    device: torch.device,
) -> int:
    """
    Cap the configured per-GPU batch size to what safely fits in GPU memory
    after FSDP sharding, then take the minimum across all ranks.

    FSDP MEMORY MODEL (different from DDP):
      With FSDP and world_size N, each rank holds:
        - 1/N of the parameters (fp16):    n_params × 2 B / N
        - 1/N of the optimizer states:     n_params × 8 B / N  (m + v in fp32)
        - 1/N of the gradient shard (fp32):n_params × 4 B / N
        Total static per rank:             n_params × 14 B / N

      However, during each block's forward/backward, the FULL block parameters
      are temporarily all-gathered. The largest block holds:
        block_params = n_params / n_layer  parameters
        all-gather peak:  block_params × 2 B  (fp16 gathered weights)

      Activations are unchanged from DDP (batch-size dependent).

    Args:
        requested:   Batch size from the config (upper bound).
        model_cfg:   GPTConfig with architecture dims.
        n_params:    Total parameter count (from model.count_parameters()).
        world_size:  Total number of FSDP ranks.
        device:      This rank's CUDA device.

    Returns:
        Safe per-GPU batch size, identical on all ranks.
    """
    props = torch.cuda.get_device_properties(device)
    total_bytes = props.total_memory
    total_gb = total_bytes / 1024 ** 3

    # ── FSDP static memory (sharded) ──────────────────────────────────────────
    # Each rank holds 1/world_size of params + optimizer states + grad shard.
    static_bytes = (n_params * 14) // world_size  # fp16 params + fp32 opt + fp32 grad

    # ── Peak all-gather buffer (one block at a time, fp16) ────────────────────
    # During forward/backward, FSDP all-gathers exactly one block's weights.
    params_per_block = n_params // model_cfg.n_layer
    allgather_bytes = params_per_block * 2  # fp16

    # ── CUDA context + allocator overhead ─────────────────────────────────────
    overhead_bytes = 500 * 1024 ** 2  # 500 MB

    # ── Per-sample activation memory (fp16) ───────────────────────────────────
    attn_bytes = model_cfg.n_head * model_cfg.seq_len * model_cfg.seq_len * 2
    mlp_bytes = (4 * model_cfg.n_embd + 2 * model_cfg.n_embd) * model_cfg.seq_len * 2
    act_per_sample = model_cfg.n_layer * (attn_bytes + mlp_bytes)

    # ── Usable memory with 25% safety margin ──────────────────────────────────
    usable_bytes = (
        total_bytes - static_bytes - allgather_bytes - overhead_bytes
    ) * 0.75
    usable_bytes = max(usable_bytes, 0)

    computed_max = max(1, int(usable_bytes / act_per_sample))
    local_safe = min(requested, computed_max)

    # ── Synchronise across all ranks — use the most constrained GPU ───────────
    local_t = torch.tensor(local_safe, dtype=torch.int32, device=device)
    dist.all_reduce(local_t, op=dist.ReduceOp.MIN)
    global_safe = int(local_t.item())

    print(
        f"[rank {dist.get_rank()}] GPU: {props.name}  "
        f"VRAM: {total_gb:.1f} GB  "
        f"static_shard: {static_bytes/1024**3:.2f} GB  "
        f"allgather_peak: {allgather_bytes/1024**2:.0f} MB  "
        f"act/sample: {act_per_sample/1024**2:.0f} MB  "
        f"computed_max_bs: {computed_max}  "
        f"config_bs: {requested}  "
        f"→ using bs={global_safe}"
    )

    return global_safe


# ── Sharded checkpoint save/load ───────────────────────────────────────────────

def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    cfg: DictConfig,
    ckpt_dir: Path,
) -> None:
    """
    Save a sharded FSDP checkpoint using torch.distributed.checkpoint (DCP).

    All ranks participate. DCP writes one shard file per rank — the full model
    is never gathered on a single GPU, so this is safe even at 1.3B params.
    rank 0 additionally writes a metadata file (step, loss, config).

    WHY NOT torch.save(model.state_dict())?
      With FSDP's default (FULL_STATE_DICT), state_dict() gathers all params
      onto rank 0 in fp32: 1.3B × 4 B ≈ 5.2 GB — OOM. DCP avoids the gather.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # DCP state: model shards + optimizer shards.
    state = {"model": model, "optimizer": optimizer}
    dcp.save(state, checkpoint_id=str(ckpt_dir))

    # Rank 0 writes human-readable metadata alongside the shard files.
    if dist.get_rank() == 0:
        meta = {
            "step": step,
            "loss": float(loss),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        with open(ckpt_dir / "meta.yaml", "w") as f:
            yaml.dump(meta, f)


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: Path,
    device: torch.device,
) -> int:
    """
    Load a sharded FSDP checkpoint. All ranks must call this together.
    Returns the step number from the checkpoint metadata.
    """
    state = {"model": model, "optimizer": optimizer}
    dcp.load(state, checkpoint_id=str(ckpt_dir))

    # Read step from metadata written by rank 0 during save.
    meta_path = ckpt_dir / "meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        step = int(meta.get("step", 0))
    else:
        step = 0

    if dist.get_rank() == 0:
        print(f"[resume] Loaded FSDP checkpoint from {ckpt_dir}  (step={step})")

    return step


# ── Main training function ─────────────────────────────────────────────────────

def train(
    cfg: DictConfig,
    rank: int,
    local_rank: int,
    world_size: int,
    resume_dir: Path | None = None,
) -> None:
    """
    FSDP training loop for the 1.3B GPT model.

    Args:
        cfg:         Merged OmegaConf config (base_config + phase3_1B).
        rank:        Global rank of this process (0 … world_size-1).
        local_rank:  Local rank within this node (= CUDA device index).
        world_size:  Total number of FSDP processes across all nodes.
        resume_dir:  If set, restore from a DCP sharded checkpoint.
    """
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    # ── Determinism ───────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig.from_omegaconf(cfg)
    # Build on CPU first — FSDP will shard and move to GPU during wrapping.
    model = GPT(model_cfg)

    if is_master:
        n_params = model.count_parameters()
        print(f"Model: {n_params / 1e9:.3f}B params")
        print(f"Rank 0 device: {device}  GPU: {torch.cuda.get_device_name(local_rank)}")

    n_params = model.count_parameters()

    # ── torch.compile — per-block, BEFORE FSDP wrapping ───────────────────────
    # FSDP wraps each TransformerBlock independently. Compiling each block
    # before FSDP means TorchInductor sees the raw compute graph without FSDP's
    # gather/scatter hooks — identical reasoning to Phase 2 (see train_ddp.py).
    # use_compile defaults to False (Amarel GCC compatibility). Enable only after
    # verifying RHEL9 GCC version supports TorchInductor (stdatomic.h requires GCC 4.9+).
    use_compile: bool = cfg.get("use_compile", False)
    if use_compile and device.type == "cuda":
        cap = torch.cuda.get_device_capability(local_rank)
        if cap[0] >= 8:
            if is_master:
                print(f"Compiling TransformerBlocks (sm_{cap[0]}{cap[1]})...")
            try:
                torch._dynamo.config.suppress_errors = True
                for i, block in enumerate(model.blocks):
                    model.blocks[i] = torch.compile(block)
                if is_master:
                    print("torch.compile: enabled (per-block)")
            except Exception as e:
                if is_master:
                    print(f"torch.compile failed ({e}), using eager mode")
        else:
            if is_master:
                print(f"torch.compile: skipped (sm_{cap[0]}{cap[1]} < sm_80)")
    else:
        if is_master:
            print(f"torch.compile: disabled (use_compile={use_compile})")

    # ── FSDP mixed precision policy ───────────────────────────────────────────
    # param_dtype=fp16: parameters are cast to fp16 during each all-gather, so
    # inter-node communication is 2× cheaper than fp32 (less bandwidth on Ethernet).
    # reduce_dtype=fp16: gradient reduce-scatter uses fp16, halving backward comms.
    # buffer_dtype=fp16: buffers (e.g. the causal attention mask) are stored fp16.
    # GradScaler is still required — FSDP MP policy does not handle loss scaling.
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # ── FSDP auto-wrap policy ─────────────────────────────────────────────────
    # min_num_params=100_000 causes FSDP to wrap every TransformerBlock (each
    # has ~50M params) as its own FSDP unit. Embedding and LM head layers are
    # smaller and remain in the root FSDP module.
    # Wrapping at the block level means:
    #   - Each all-gather/reduce-scatter moves ~50M params (~100 MB fp16).
    #   - Blocks can be prefetched: while block N is running forward, block N+1's
    #     parameters are being all-gathered in the background.
    wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000)

    # ── Wrap with FSDP ────────────────────────────────────────────────────────
    # FULL_SHARD = ZeRO-3: shard params + grads + optimizer states across all ranks.
    # device_id: FSDP moves each shard to the correct GPU during wrapping.
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
    )

    # ── Gradient (activation) checkpointing ───────────────────────────────────
    # Applied AFTER FSDP wrapping — wraps each FSDP'd TransformerBlock's forward
    # with torch.utils.checkpoint.checkpoint(). During backward, activations are
    # recomputed from the block inputs rather than stored. Reduces peak activation
    # memory by ~√(n_layers) at the cost of ~30% extra compute per step.
    # NO_REENTRANT avoids conflicts with FSDP's own reentrant backward hooks.
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda m: isinstance(m, TransformerBlock),
    )

    if is_master:
        print("Gradient checkpointing: enabled (NO_REENTRANT, per TransformerBlock)")

    # ── Auto batch size ───────────────────────────────────────────────────────
    # Must run after model is wrapped with FSDP (static shard is allocated) but
    # before the DataLoader is built. model_cfg and n_params are captured above.
    safe_bs = auto_batch_size(cfg.batch_size, model_cfg, n_params, world_size, device)
    if safe_bs != cfg.batch_size:
        if is_master:
            print(
                f"[auto_batch_size] Reduced batch_size {cfg.batch_size} → {safe_bs} "
                f"to fit GPU memory."
            )
        cfg = OmegaConf.merge(cfg, {"batch_size": safe_bs})

    # ── Optimizer with linear LR scaling ──────────────────────────────────────
    # Linear scaling rule: effective_lr = base_lr × world_size.
    # With world_size=16: effective_lr = 6e-5 × 16 = 9.6e-4.
    # build_optimizer is called on model (FSDP-wrapped); it calls named_parameters()
    # which FSDP implements to iterate over this rank's sharded parameters.
    scaled_lr = cfg.learning_rate * world_size
    cfg_scaled = OmegaConf.merge(cfg, {"learning_rate": scaled_lr})
    optimizer = build_optimizer(model, cfg_scaled)

    if is_master:
        print(
            f"LR: base={cfg.learning_rate:.2e} × world_size={world_size}"
            f" → effective={scaled_lr:.2e}"
        )
        print(
            f"Effective batch size: {cfg.batch_size} × {world_size}"
            f" = {cfg.batch_size * world_size} sequences/step"
        )

    # ── GradScaler ────────────────────────────────────────────────────────────
    # Each rank manages its own scaler independently. FSDP's reduce-scatter
    # averages the fp16-scaled gradients; scaling and unscaling are symmetric
    # across ranks so the effective update is identical to single-GPU.
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if resume_dir is not None:
        # All ranks load their own shards simultaneously (DCP coordinates this).
        start_step = load_fsdp_checkpoint(model, optimizer, resume_dir, device)

    # Broadcast start_step from rank 0 to confirm all ranks agree.
    start_step_t = torch.tensor(start_step, device=device)
    dist.broadcast(start_step_t, src=0)
    start_step = int(start_step_t.item())

    # ── Overfit test (rank 0 only) ─────────────────────────────────────────────
    # Tests that the GPT code paths work (forward, backward, optimizer) using a
    # tiny proxy model. Does not touch the FSDP-wrapped 1.3B model.
    # run_overfit_test only reads model.config.vocab_size to build a nano proxy;
    # we pass a lightweight namespace rather than the FSDP wrapper to avoid
    # relying on FSDP's undocumented __getattr__ delegation.
    if start_step == 0:
        if is_master:
            run_overfit_test(types.SimpleNamespace(config=model_cfg), device)
        dist.barrier()

    # ── Data ──────────────────────────────────────────────────────────────────
    assert os.path.exists(cfg.data_path), (
        f"Training data not found: {cfg.data_path}\n"
        "Run: wget .../TinyStoriesV2-GPT4-train.txt -P $SCRATCH/data/tinystories/"
    )

    train_loader = build_fsdp_dataloader(
        file_path=cfg.data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=2,
    )

    # ── Checkpoint paths (all ranks need them for DCP) ────────────────────────
    # Broadcast a Unix timestamp integer from rank 0 so all ranks format the
    # same directory name. Avoids char-by-char encoding and is length-agnostic.
    ts_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    if is_master:
        ts_tensor[0] = int(time.time())
    dist.broadcast(ts_tensor, src=0)
    timestamp = datetime.fromtimestamp(int(ts_tensor[0].item())).strftime("%Y%m%d_%H%M%S")

    ckpt_root = Path(
        cfg.get("checkpoint_dir", os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    )
    run_dir = ckpt_root / f"run_{timestamp}"
    best_ckpt_dir = run_dir / "best"
    best_loss = float("inf")
    save_best_only: bool = cfg.get("save_best_only", False)

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
        if save_best_only:
            print("Checkpoint policy: best-only (no step_* or _final saves)")

    while step < cfg.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # LR schedule (cosine with warmup using the linearly scaled LR).
        lr = get_lr(step, scaled_lr, cfg.warmup_steps, cfg.max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward pass ──────────────────────────────────────────────────────
        # FSDP's MixedPrecision handles fp16 casting during all-gather.
        # We do NOT use torch.autocast here — FSDP MP replaces it.
        # The loss is computed in fp32 (FSDP keeps the output in compute_dtype).
        _, loss = model(x, y)

        # ── Backward pass ─────────────────────────────────────────────────────
        # FSDP's reduce-scatter runs during backward — by the time backward()
        # returns, each rank holds the reduced gradient shard for its params.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.grad_clip
        )

        scaler.step(optimizer)
        scaler.update()

        step += 1

        # ── Per-rank memory logging ────────────────────────────────────────────
        # Log every 100 steps so GPU imbalance is visible early in the run.
        # All ranks print — cross-rank comparison shows whether sharding is even.
        if step % 100 == 0:
            allocated_gb = torch.cuda.memory_allocated(device) / 1024 ** 3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024 ** 3
            print(
                f"[rank {rank}] step {step:6d} | "
                f"mem alloc {allocated_gb:.2f} GB | reserved {reserved_gb:.2f} GB"
            )

        # ── Loss / throughput logging (rank 0 only) ───────────────────────────
        if is_master and step % cfg.log_interval == 0:
            elapsed = time.time() - t0
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

        # ── Periodic step checkpoints (optional) ────────────────────────────────
        # When save_best_only=true (phase3_1B.yaml), skip step_* saves to avoid
        # filling $SCRATCH — each DCP checkpoint is ~8 GB × 100 steps ≈ 800 GB.
        if not save_best_only and step % cfg.save_interval == 0:
            save_fsdp_checkpoint(
                model, optimizer, step, loss.item(), cfg,
                run_dir / f"step_{step:07d}",
            )
            dist.barrier()

        # ── Save-best checkpoint ───────────────────────────────────────────────
        # CRITICAL: dcp.save() inside save_fsdp_checkpoint is a collective —
        # ALL ranks must call it together. Each rank has its own mini-batch loss,
        # so evaluating `current_loss < best_loss` independently per rank would
        # cause split decisions → some ranks call dcp.save(), others skip → deadlock.
        # Fix: rank 0 makes the decision and broadcasts it (int32 tensor) so all
        # ranks enter or skip the DCP collective together.
        current_loss = loss.item()
        save_best_t = torch.tensor(
            int(is_master and current_loss < best_loss),
            dtype=torch.int32, device=device,
        )
        dist.broadcast(save_best_t, src=0)

        if save_best_t.item():
            if is_master:
                best_loss = current_loss
                if best_ckpt_dir.exists():
                    shutil.rmtree(best_ckpt_dir)
            dist.barrier()  # ensure rmtree completes before new write
            save_fsdp_checkpoint(
                model, optimizer, step, current_loss, cfg, best_ckpt_dir,
            )
        dist.barrier()

    # ── Final checkpoint (optional) ───────────────────────────────────────────
    if not save_best_only:
        save_fsdp_checkpoint(
            model, optimizer, step, loss.item(), cfg,
            run_dir / f"step_{step:07d}_final",
        )
    if is_master:
        ckpt_note = f"{run_dir}/best" if save_best_only else str(run_dir)
        print(f"\nTraining complete. Checkpoints: {ckpt_note}")

    dist.barrier()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-node FSDP GPT training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Phase config YAML (e.g. configs/phase3_1B.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="DCP checkpoint directory to resume from",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override data_path from config",
    )
    return parser.parse_args()


def main() -> None:
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
        # Force GC before NCCL teardown so FSDP C++ destructors run in the
        # correct order (same fix as Phase 2 — see train_ddp.py for explanation).
        gc.collect()
        cleanup_distributed()


if __name__ == "__main__":
    main()
