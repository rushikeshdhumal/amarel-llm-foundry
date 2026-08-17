"""
export_fsdp_checkpoint.py — Gather FSDP DCP shards into a single model.pt for inference.

All ranks must participate in dcp.load(); rank 0 writes model.pt + config.yaml to
<checkpoint_dir>_export/.

Export wraps FSDP in fp32 (no MixedPrecision). Training's fp16 policy makes
summon/FULL_STATE_DICT return compute-dtype copies instead of the fp32 master
weights, which evals as ~random (val loss 7.5 vs train 0.58). DCP shards are
fp32; load them into an fp32 FSDP tree, then gather with get_model_state_dict.

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
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from src.generate import canonicalize_state_dict_key
from src.model import GPT, GPTConfig, TransformerBlock
from src.train_fsdp import (
    build_optimizer,
    cleanup_distributed,
    load_fsdp_checkpoint,
    setup_distributed,
)


def log_local_param_stats(model: torch.nn.Module, tag: str, rank: int) -> None:
    """Print a few local parameter tensors so we can see if dcp.load changed them."""
    printed = 0
    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue
        tensor = param.detach().float()
        print(
            f"[export][rank {rank}] {tag} {name}: "
            f"shape={tuple(param.shape)} dtype={param.dtype} "
            f"mean={tensor.mean().item():.5f} std={tensor.std().item():.5f}"
        )
        printed += 1
        if printed >= 3:
            return


def build_fsdp_model(cfg: OmegaConf, device: torch.device, world_size: int) -> tuple[FSDP, torch.optim.Optimizer]:
    """Rebuild FSDP with the same wrap/shard policy as training, but fp32 params."""
    model_cfg = GPTConfig.from_omegaconf(cfg)
    model = GPT(model_cfg)

    wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000)

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
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


def collapse_wrapper_keys(raw_state: dict, is_master: bool) -> dict:
    """
    Strip wrapper prefixes. If several raw keys collapse to the same name,
    keep the largest tensor (full param over a shard / stale view).
    """
    grouped: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for key, value in raw_state.items():
        grouped.setdefault(canonicalize_state_dict_key(key), []).append((key, value))

    collapsed: dict = {}
    for canon, items in grouped.items():
        if len(items) > 1 and is_master:
            shapes = [(src, tuple(tensor.shape)) for src, tensor in items]
            print(f"[export] Key collision for {canon}: {shapes}")
        items_sorted = sorted(items, key=lambda kv: kv[1].numel(), reverse=True)
        collapsed[canon] = items_sorted[0][1].detach().cpu().contiguous().clone()
    return collapsed


def gather_full_state(model: FSDP) -> dict:
    """
    All-gather a full state dict without mutating the FSDP module tree.

    get_model_state_dict(full_state_dict=True) is the PyTorch 2.x API for a
    plain (unwrapped) GPT state dict. All ranks must call it.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


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
        print("[export] FSDP wrap is fp32 (no MixedPrecision) so gather uses master weights.")

    log_local_param_stats(model, "before_load", rank)
    dist.barrier()
    load_fsdp_checkpoint(model, optimizer, ckpt_dir, device)
    dist.barrier()
    log_local_param_stats(model, "after_load", rank)
    dist.barrier()
    if is_master:
        print("[export] DCP load complete. Gathering full parameters...")

    full_state = gather_full_state(model)
    dist.barrier()

    if is_master:
        export_dir = Path(f"{ckpt_dir}_export")
        export_dir.mkdir(parents=True, exist_ok=True)

        print(f"[export] Raw gathered keys (first 12): {list(full_state.keys())[:12]}")
        print(f"[export] Raw gathered key count: {len(full_state)}")
        plain_state = collapse_wrapper_keys(full_state, is_master=True)
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

        tok = plain_state["tok_emb.weight"].float()
        print(
            f"[export] tok_emb.weight shape={tuple(tok.shape)} "
            f"mean={tok.mean().item():.5f} std={tok.std().item():.5f} "
            f"(init is std≈0.02; a trained embedding is typically larger)"
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
