"""
export_fsdp_checkpoint.py — Gather FSDP DCP shards into a single model.pt for inference.

All ranks must participate in dcp.load(); rank 0 writes model.pt + config.yaml to
<checkpoint_dir>_export/. Parameters are gathered with FSDP.summon_full_params
(not FULL_STATE_DICT through activation-checkpoint wrappers).

Launch with the same world_size and node layout as the training job that wrote the
checkpoint (default Phase 3: 4 nodes × 4 GPUs = 16).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from src.generate import strip_wrapper_prefixes
from src.model import GPTConfig, TransformerBlock
from src.train_fsdp import (
    build_optimizer,
    cleanup_distributed,
    load_fsdp_checkpoint,
    setup_distributed,
)


def build_fsdp_model(cfg: OmegaConf, device: torch.device, world_size: int) -> tuple[FSDP, torch.optim.Optimizer]:
    """Rebuild the FSDP-wrapped model with the same policy as train_fsdp.py."""
    model_cfg = GPTConfig.from_omegaconf(cfg)
    model = GPT(model_cfg)

    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000)

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
    )

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda m: isinstance(m, TransformerBlock),
    )

    scaled_lr = cfg.learning_rate * world_size
    cfg_scaled = OmegaConf.merge(cfg, {"learning_rate": scaled_lr})
    optimizer = build_optimizer(model, cfg_scaled)

    return model, optimizer


def load_checkpoint_config(ckpt_dir: Path) -> tuple[OmegaConf, int, float]:
    """Read merged training config and metadata from a DCP checkpoint directory."""
    meta_path = ckpt_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No meta.yaml in {ckpt_dir}. Expected an FSDP DCP checkpoint directory."
        )

    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    if "config" not in meta:
        raise KeyError(f"meta.yaml in {ckpt_dir} missing 'config' field.")

    cfg = OmegaConf.create(meta["config"])
    step = int(meta.get("step", 0))
    loss = float(meta.get("loss", 0.0))
    return cfg, step, loss


def count_unique_params(state: dict) -> int:
    """Count parameter elements, skipping reconstructed buffers and tied lm_head."""
    skip_lm_head = "tok_emb.weight" in state and "lm_head.weight" in state
    total = 0
    for key, tensor in state.items():
        if key.endswith(".mask"):
            continue
        if skip_lm_head and key == "lm_head.weight":
            continue
        total += tensor.numel()
    return total


def gather_full_state(model: FSDP, is_master: bool) -> dict:
    """
    All-gather full parameters without mutating the FSDP module tree.

    summon_full_params talks to FSDP handles directly. That avoids FULL_STATE_DICT
    walking through CheckpointWrapper (wrong values) and avoids unwrapping those
    wrappers (NCCL deadlock).
    """
    full_state: dict = {}
    with FSDP.summon_full_params(
        model,
        recurse=True,
        writeback=False,
        rank0_only=True,
        offload_to_cpu=True,
        with_grads=False,
    ):
        if is_master:
            for name, param in model.named_parameters():
                full_state[name] = param.detach().cpu().contiguous().clone()
            for name, buf in model.named_buffers():
                full_state[name] = buf.detach().cpu().contiguous().clone()
    return full_state


def export_checkpoint(
    ckpt_dir: Path,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    """Load DCP shards collectively and write a single-GPU export on rank 0."""
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0

    cfg, step, loss = load_checkpoint_config(ckpt_dir)
    model, optimizer = build_fsdp_model(cfg, device, world_size)

    if is_master:
        print(f"[export] Loading DCP checkpoint from {ckpt_dir} (step={step}, loss={loss:.4f})")

    load_fsdp_checkpoint(model, optimizer, ckpt_dir, device)
    dist.barrier()
    if is_master:
        print("[export] DCP load complete. Gathering full parameters...")

    full_state = gather_full_state(model, is_master)
    dist.barrier()

    if is_master:
        export_dir = Path(f"{ckpt_dir}_export")
        export_dir.mkdir(parents=True, exist_ok=True)

        plain_state = strip_wrapper_prefixes(full_state)
        required = [
            "tok_emb.weight",
            "pos_emb.weight",
            "ln_f.weight",
            "blocks.0.ln1.weight",
            "lm_head.weight",
        ]
        missing = [key for key in required if key not in plain_state]
        if missing:
            sample = list(plain_state.keys())[:12]
            raise RuntimeError(
                f"Gathered state_dict missing {missing}. Sample keys: {sample}"
            )

        weights_path = export_dir / "model.pt"
        print(f"[export] Writing {weights_path} ...")
        torch.save(plain_state, weights_path)

        save_cfg = OmegaConf.to_container(cfg, resolve=True)
        save_cfg["_step"] = step
        save_cfg["_loss"] = loss
        OmegaConf.save(OmegaConf.create(save_cfg), export_dir / "config.yaml")

        unique_params = count_unique_params(plain_state)
        print(f"[export] State dict keys (first 5): {list(plain_state.keys())[:5]}")
        print(f"[export] Wrote {unique_params / 1e9:.3f}B unique params → {weights_path}")
        print(f"[export] Config → {export_dir / 'config.yaml'}")
        print("[export] Done. Use this directory with generate.py and eval.py.")

    dist.barrier()
    if is_master:
        print("[export] All ranks finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FSDP DCP checkpoint to model.pt")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="DCP checkpoint directory (contains meta.yaml + shard files)",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    rank, local_rank, world_size = setup_distributed()
    args = parse_args()
    ckpt_dir = Path(args.checkpoint)

    if rank == 0 and not (ckpt_dir / "meta.yaml").exists():
        raise FileNotFoundError(f"Not an FSDP DCP checkpoint: {ckpt_dir}")

    try:
        export_checkpoint(ckpt_dir, rank, local_rank, world_size)
    finally:
        gc.collect()
        cleanup_distributed()


if __name__ == "__main__":
    main()
