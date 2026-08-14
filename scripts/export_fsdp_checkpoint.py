"""
export_fsdp_checkpoint.py — Gather FSDP DCP shards into a single model.pt for inference.

All ranks must participate in dcp.load(); rank 0 writes model.pt + config.yaml to
<checkpoint_dir>_export/ using FullStateDictConfig(offload_to_cpu=True, rank0_only=True).

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
import torch.distributed.checkpoint as dcp
import yaml
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from src.generate import strip_wrapper_prefixes
from src.model import GPT, GPTConfig, TransformerBlock
from src.train_fsdp import (
    build_optimizer,
    cleanup_distributed,
    load_fsdp_checkpoint,
    setup_distributed,
)


def unwrap_activation_checkpointing(module: torch.nn.Module) -> None:
    """
    Replace CheckpointWrapper modules with their inner module.

    apply_activation_checkpointing is required so dcp.load() sees the same tree
    as training, but FULL_STATE_DICT gather through those wrappers can produce
    correctly shaped tensors with wrong values. Unwrap after load, before gather.
    """
    if isinstance(module, FSDP):
        inner = module._fsdp_wrapped_module
        wrapped = getattr(inner, "_checkpoint_wrapped_module", None)
        if wrapped is not None:
            module._fsdp_wrapped_module = wrapped
            unwrap_activation_checkpointing(wrapped)
        else:
            unwrap_activation_checkpointing(inner)
        return

    if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
        for idx, child in enumerate(list(module)):
            wrapped = getattr(child, "_checkpoint_wrapped_module", None)
            if wrapped is not None:
                module[idx] = wrapped
                unwrap_activation_checkpointing(wrapped)
            else:
                unwrap_activation_checkpointing(child)
        return

    if isinstance(module, torch.nn.ModuleDict):
        for key, child in list(module.items()):
            wrapped = getattr(child, "_checkpoint_wrapped_module", None)
            if wrapped is not None:
                module[key] = wrapped
                unwrap_activation_checkpointing(wrapped)
            else:
                unwrap_activation_checkpointing(child)
        return

    for name, child in list(module.named_children()):
        wrapped = getattr(child, "_checkpoint_wrapped_module", None)
        if wrapped is not None:
            setattr(module, name, wrapped)
            unwrap_activation_checkpointing(wrapped)
        else:
            unwrap_activation_checkpointing(child)


def to_plain_gpt_state_dict(raw_state: dict, cfg: OmegaConf) -> tuple[dict, int]:
    """Strip wrapper prefixes and reload into a plain GPT so keys match generate.py."""
    cleaned = strip_wrapper_prefixes(raw_state)
    ref = GPT(GPTConfig.from_omegaconf(cfg))
    incompatible = ref.load_state_dict(cleaned, strict=False)
    missing_params = [k for k in incompatible.missing_keys if not k.endswith(".mask")]
    if missing_params:
        raise RuntimeError(
            "Exported state_dict is missing GPT parameter keys "
            f"(showing up to 8): {missing_params[:8]}"
        )
    ref.lm_head.weight = ref.tok_emb.weight
    n_params = ref.count_parameters()
    plain = {k: v.detach().cpu().contiguous() for k, v in ref.state_dict().items()}
    return plain, n_params


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

    unwrap_activation_checkpointing(model)
    dist.barrier()

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_state = model.state_dict()

    if is_master:
        export_dir = Path(f"{ckpt_dir}_export")
        export_dir.mkdir(parents=True, exist_ok=True)

        plain_state, unique_params = to_plain_gpt_state_dict(full_state, cfg)
        weights_path = export_dir / "model.pt"
        torch.save(plain_state, weights_path)

        save_cfg = OmegaConf.to_container(cfg, resolve=True)
        save_cfg["_step"] = step
        save_cfg["_loss"] = loss
        OmegaConf.save(OmegaConf.create(save_cfg), export_dir / "config.yaml")

        sample_keys = list(plain_state.keys())[:5]
        print(f"[export] State dict keys (first 5): {sample_keys}")
        print(f"[export] Wrote {unique_params / 1e9:.3f}B unique params → {weights_path}")
        print(f"[export] Config → {export_dir / 'config.yaml'}")
        print("[export] Done. Use this directory with generate.py and eval.py.")

    dist.barrier()


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
