"""
dist_hello.py – Phase 00 distributed sanity check.

Initializes torch.distributed and prints rank / world-size / hostname
so we can confirm NCCL is wired up correctly across Amarel nodes.

Usage (via torchrun – see slurm/hello_*.sh):
    torchrun --nproc_per_node=<GPUS> --nnodes=<NODES> \
             --node_rank=<RANK> --master_addr=<ADDR> --master_port=<PORT> \
             src/dist_hello.py [--backend nccl|gloo]
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist


# ── NCCL safety: must be set BEFORE dist.init_process_group ──────────────────
# PyTorch 2.x renamed this variable; set both for backwards compatibility.
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def resolve_master_addr() -> str:
    """
    Derive MASTER_ADDR from the first host in SLURM_NODELIST.

    Amarel sets SLURM_NODELIST to a compact form like 'node[001-002]'.
    `scontrol show hostnames` expands it, but that requires a subprocess.
    torchrun already receives --master_addr from the SLURM script, so this
    function is a fallback guard used only when the env var is unset.
    """
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return "127.0.0.1"

    # Extract the first hostname from compact SLURM notation.
    # Simple heuristic: strip bracket ranges, take first token.
    import re
    first = re.split(r"[,\[]", nodelist)[0]
    try:
        return socket.gethostbyname(first)
    except socket.gaierror:
        return first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed hello-world for Amarel")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="torch.distributed backend (nccl for GPU, gloo for CPU fallback)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)

    # torchrun exports RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    # Guard: if MASTER_ADDR is missing (manual launch), derive it from SLURM.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = resolve_master_addr()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))

    # Pin this process to its GPU BEFORE init_process_group.
    # NCCL requires the device assignment to be known at group creation time;
    # setting it afterwards triggers "devices unknown" warnings and can cause
    # ncclInvalidUsage on the first collective (barrier/all-reduce).
    if args.backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")
    else:
        device_id = None

    # Pass device_id so PyTorch can associate each rank with its GPU at init
    # time, eliminating the barrier "devices unknown" warning.
    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        device_id=device_id,
    )

    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()
    hostname: str = socket.gethostname()

    if device_id is not None:
        device_name: str = torch.cuda.get_device_name(local_rank)
    else:
        device_name = "cpu"

    # Barrier ensures all ranks have initialised before printing.
    dist.barrier()

    # Each rank prints its own line; rank 0 also prints a header/footer.
    if rank == 0:
        print(f"{'─' * 60}", flush=True)
        print(f"  torch.distributed hello-world  |  backend={args.backend}", flush=True)
        print(f"{'─' * 60}", flush=True)

    dist.barrier()

    print(
        f"[Rank {rank}/{world_size}] "
        f"Host: {hostname}  "
        f"local_rank={local_rank}  "
        f"device={device_name}",
        flush=True,
    )

    dist.barrier()

    if rank == 0:
        print(f"{'─' * 60}", flush=True)
        print("All ranks healthy. Destroying process group.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
