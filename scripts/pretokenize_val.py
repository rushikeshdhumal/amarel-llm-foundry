"""
pretokenize_val.py — Tokenize TinyStories validation split to val.bin (uint16 memmap).

Matches training tokenization in src/data_utils.py:
  line-by-line read, Tokenizer.encode(line, add_eot=False), concatenate stream.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from src.tokenizer import Tokenizer

VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStoriesV2-GPT4-valid.txt"
)
DEFAULT_VALID_NAME = "TinyStoriesV2-GPT4-valid.txt"


def tokenize_file(file_path: Path) -> list[int]:
    """Tokenize a TinyStories .txt file into a contiguous token ID stream."""
    tokenizer = Tokenizer()
    tokens: list[int] = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.extend(tokenizer.encode(line, add_eot=False))

    return tokens


def write_val_bin(tokens: list[int], out_dir: Path, source_path: Path) -> None:
    """Write uint16 memmap and metadata YAML."""
    out_dir.mkdir(parents=True, exist_ok=True)
    val_bin = out_dir / "val.bin"
    val_meta = out_dir / "val_meta.yaml"

    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(val_bin)

    meta = {
        "val_tokens": int(len(tokens)),
        "source": str(source_path),
        "source_type": "official_valid_split",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dtype": "uint16",
        "encoding": "gpt2_bpe",
        "add_eot": False,
    }
    with open(val_meta, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"[pretokenize_val] Wrote {len(tokens):,} tokens → {val_bin}")
    print(f"[pretokenize_val] Metadata → {val_meta}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize TinyStories validation split.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Path to valid .txt (default: $SCRATCH/data/tinystories/{DEFAULT_VALID_NAME})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for val.bin (default: $SCRATCH/data/tinystories/tokenized/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}")
    default_input = Path(scratch) / "data" / "tinystories" / DEFAULT_VALID_NAME
    default_output = Path(scratch) / "data" / "tinystories" / "tokenized"

    input_path = Path(args.input) if args.input else default_input
    output_dir = Path(args.output_dir) if args.output_dir else default_output

    if not input_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {input_path}\n"
            f"Download with:\n"
            f"  wget {VALID_URL} -P {input_path.parent}/"
        )

    tokens = tokenize_file(input_path)
    if len(tokens) < 1025:
        raise ValueError(
            f"Only {len(tokens)} tokens in {input_path} — need at least seq_len+1 for eval."
        )

    write_val_bin(tokens, output_dir, input_path)


if __name__ == "__main__":
    main()
