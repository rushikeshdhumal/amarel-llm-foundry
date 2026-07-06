"""
eval.py — Evaluate a trained GPT checkpoint on pretokenized TinyStories validation data.

Metrics: val loss (mean CE), perplexity = exp(loss), bits_per_token = loss / ln(2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generate import load_model_from_checkpoint
from src.model import GPT


def resolve_path(path_str: str) -> Path:
    """Expand environment variables in a config path."""
    return Path(os.path.expandvars(path_str)).expanduser()


def load_val_tokens(val_path: Path) -> np.ndarray:
    """Load uint16 memmap validation tokens."""
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation memmap not found: {val_path}\n"
            "Run: sbatch slurm/pretokenize_val.sh"
        )
    return np.memmap(val_path, dtype=np.uint16, mode="r")


@torch.no_grad()
def evaluate_gpt_on_val(
    model: GPT,
    tokens: np.ndarray,
    seq_len: int,
    device: torch.device,
    max_tokens: int,
    batch_size: int,
) -> dict[str, float]:
    """
    Compute mean cross-entropy loss over non-overlapping (x, y) windows.

    Windowing matches training: x = chunk[:seq_len], y = chunk[1:seq_len+1],
    stride = seq_len + 1.
    """
    model.eval()
    stride = seq_len + 1
    n_tokens = len(tokens)

    if n_tokens < stride:
        raise ValueError(f"Need at least {stride} val tokens, got {n_tokens}.")

    starts: list[int] = []
    tokens_evaluated = 0
    for start in range(0, n_tokens - seq_len, stride):
        if tokens_evaluated + seq_len > max_tokens:
            break
        starts.append(start)
        tokens_evaluated += seq_len

    if not starts:
        raise ValueError("No evaluation windows fit within max_tokens limit.")

    total_loss = 0.0
    total_windows = 0

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        xs: list[list[int]] = []
        ys: list[list[int]] = []

        for start in batch_starts:
            chunk = tokens[start : start + stride]
            xs.append(chunk[:seq_len].tolist())
            ys.append(chunk[1 : seq_len + 1].tolist())

        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)

        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Model returned no loss — targets may be missing.")

        total_loss += loss.item() * len(batch_starts)
        total_windows += len(batch_starts)

    val_loss = total_loss / total_windows
    perplexity = math.exp(val_loss)
    bits_per_token = val_loss / math.log(2)

    return {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "bits_per_token": bits_per_token,
        "num_windows": float(total_windows),
        "tokens_evaluated": float(total_windows * seq_len),
        "seq_len": float(seq_len),
    }


def write_results(
    results: dict,
    output_dir: Path,
    checkpoint_dir: Path,
    name: str | None,
) -> Path:
    """Write eval metrics to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = name if name else checkpoint_dir.name
    out_path = output_dir / f"{out_name}.json"

    payload = {
        "checkpoint": str(checkpoint_dir),
        "name": out_name,
        **results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[eval] Wrote results → {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT checkpoint on val.bin")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory with model.pt + config.yaml",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to val.bin (default: $SCRATCH/data/tinystories/tokenized/val.bin)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000,
        help="Cap tokens evaluated (default: 1M)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Windows per forward pass (use 1 for 1.3B on 32G GPU)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="JSON output directory (default: $SCRATCH/eval_results/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output JSON basename (default: checkpoint dir name)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cuda or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}")
    val_path = resolve_path(
        args.val_path or f"{scratch}/data/tinystories/tokenized/val.bin"
    )
    output_dir = resolve_path(args.output_dir or f"{scratch}/eval_results")
    checkpoint_dir = resolve_path(args.checkpoint)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[eval] Device: {device}")
    print(f"[eval] Val data: {val_path}")

    tokens = load_val_tokens(val_path)
    model, cfg = load_model_from_checkpoint(checkpoint_dir, device)
    seq_len = int(cfg.get("seq_len", 1024))

    results = evaluate_gpt_on_val(
        model=model,
        tokens=tokens,
        seq_len=seq_len,
        device=device,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    print(
        f"[eval] val_loss={results['val_loss']:.4f}  "
        f"ppl={results['perplexity']:.2f}  "
        f"bpt={results['bits_per_token']:.4f}  "
        f"windows={int(results['num_windows'])}"
    )

    write_results(results, output_dir, checkpoint_dir, args.name)


if __name__ == "__main__":
    main()
