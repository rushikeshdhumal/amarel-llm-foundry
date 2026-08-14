"""
benchmark.py — Build a comparison report: project checkpoints + Hugging Face GPT-2 baselines.

transformers is used ONLY here for GPT-2 baselines (never for training — see GLOBAL.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval import evaluate_gpt_on_val, load_val_tokens, resolve_path
from src.generate import generate_text, load_model_from_checkpoint

COMPARISON_FOOTNOTE = (
    "Our models: TinyStories pretrain. GPT-2: WebText. Same GPT-2 BPE tokenizer; "
    "Phase 3 vs gpt2-xl is param-count approximate, not architecture-matched. "
    "Not matched on training tokens or corpus."
)


def load_eval_json(path: Path) -> dict | None:
    """Load a prior eval JSON result if it exists."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_hf_gpt2_on_val(
    model_name: str,
    tokens: np.ndarray,
    seq_len: int,
    device: torch.device,
    max_tokens: int,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate a Hugging Face GPT-2 model on the same val.bin windowing."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    stride = seq_len + 1
    n_tokens = len(tokens)

    starts: list[int] = []
    tokens_evaluated = 0
    for start in range(0, n_tokens - seq_len, stride):
        if tokens_evaluated + seq_len > max_tokens:
            break
        starts.append(start)
        tokens_evaluated += seq_len

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

        input_ids = torch.tensor(xs, dtype=torch.long, device=device)
        labels = torch.tensor(ys, dtype=torch.long, device=device)

        # Compute CE ourselves with the same already-shifted (x, y) windows as
        # eval.py. Hugging Face GPT-2 shifts labels internally, so passing y as
        # `labels=` would double-shift and inflate perplexity.
        logits = model(input_ids=input_ids).logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        total_loss += loss.item() * len(batch_starts)
        total_windows += len(batch_starts)

    val_loss = total_loss / total_windows

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "bits_per_token": val_loss / math.log(2),
        "num_windows": float(total_windows),
        "tokens_evaluated": float(total_windows * seq_len),
        "seq_len": float(seq_len),
    }


@torch.no_grad()
def generate_hf_samples(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[tuple[str, str]]:
    """Generate completions from a Hugging Face GPT-2 model (one prompt at a time)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    samples: list[tuple[str, str]] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        completion = full[len(prompt) :]
        samples.append((prompt, completion))

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return samples


def generate_project_samples(
    checkpoint_dir: Path,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[tuple[str, str]]:
    """Generate completions from a project GPT checkpoint."""
    model, _ = load_model_from_checkpoint(checkpoint_dir, device)
    enc = tiktoken.get_encoding("gpt2")

    samples: list[tuple[str, str]] = []
    for prompt in prompts:
        completion = generate_text(
            model=model,
            enc=enc,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        samples.append((prompt, completion))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return samples


def format_metrics_row(name: str, metrics: dict) -> str:
    """Format one markdown table row."""
    return (
        f"| {name} | {metrics['val_loss']:.4f} | "
        f"{metrics['perplexity']:.2f} | {metrics['bits_per_token']:.4f} |"
    )


def write_report(
    out_path: Path,
    footnote: str,
    project_metrics: list[tuple[str, dict]],
    hf_metrics: list[tuple[str, dict]],
    project_samples: list[tuple[str, list[tuple[str, str]]]],
    hf_samples: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    """Write markdown benchmark report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Amarel LLM Foundry — Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Comparison footnote",
        "",
        f"*{footnote.strip()}*",
        "",
        "## Validation perplexity (TinyStories val.bin)",
        "",
        "| Model | Val loss | Perplexity | Bits/token |",
        "| :--- | ---: | ---: | ---: |",
    ]

    for name, metrics in project_metrics:
        lines.append(format_metrics_row(name, metrics))

    for name, metrics in hf_metrics:
        lines.append(format_metrics_row(name, metrics))

    lines.extend(["", "## Generation samples", ""])

    for name, samples in project_samples:
        lines.append(f"### {name} (project checkpoint)")
        lines.append("")
        for prompt, completion in samples:
            lines.append(f"**Prompt:** {prompt}")
            lines.append("")
            lines.append(f"> {prompt}{completion}")
            lines.append("")

    for name, samples in hf_samples:
        lines.append(f"### {name} (Hugging Face baseline)")
        lines.append("")
        for prompt, completion in samples:
            lines.append(f"**Prompt:** {prompt}")
            lines.append("")
            lines.append(f"> {prompt}{completion}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[benchmark] Wrote report → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark project models vs GPT-2 baselines")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_suite.yaml",
        help="Eval suite config (prompts, val path, GPT-2 models)",
    )
    parser.add_argument(
        "--checkpoints_config",
        type=str,
        default="configs/eval_checkpoints.yaml",
        help="Project checkpoint paths",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000,
        help="Cap val tokens evaluated per model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Eval batch size (use 1 for 1.3B / gpt2-xl on 32G GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cuda or cpu",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation samples (perplexity table only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suite_cfg = OmegaConf.load(args.config)
    ckpt_cfg = OmegaConf.load(args.checkpoints_config)

    val_path = resolve_path(str(suite_cfg.val_path))
    eval_results_dir = resolve_path(str(suite_cfg.get("eval_results_dir", os.environ.get("SCRATCH", "") + "/eval_results")))
    benchmark_results_dir = resolve_path(str(suite_cfg.get("benchmark_results_dir", os.environ.get("SCRATCH", "") + "/benchmark_results")))
    report_path = benchmark_results_dir / "report.md"

    footnote = str(suite_cfg.get("comparison_footnote", COMPARISON_FOOTNOTE)).strip()
    prompts = list(suite_cfg.prompts)
    sampling = suite_cfg.sampling
    max_new_tokens = int(sampling.max_new_tokens)
    temperature = float(sampling.temperature)
    top_k = int(sampling.top_k)
    gpt2_models = list(suite_cfg.gpt2_models)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[benchmark] Device: {device}")
    print(f"[benchmark] Val data: {val_path}")

    tokens = load_val_tokens(val_path)
    seq_len = 1024

    project_metrics: list[tuple[str, dict]] = []
    project_samples: list[tuple[str, list[tuple[str, str]]]] = []

    for entry in ckpt_cfg.checkpoints:
        name = str(entry.name)
        ckpt_path = resolve_path(str(entry.path))

        if "<" in str(entry.path):
            print(f"[benchmark] Skipping {name}: placeholder path in eval_checkpoints.yaml")
            continue

        if not ckpt_path.exists() or not (ckpt_path / "model.pt").exists():
            print(f"[benchmark] Skipping {name}: checkpoint not found at {ckpt_path}")
            continue

        json_path = eval_results_dir / f"{name}.json"
        metrics = load_eval_json(json_path)

        if metrics is None:
            print(f"[benchmark] No JSON for {name} — running eval inline...")
            model, cfg = load_model_from_checkpoint(ckpt_path, device)
            seq_len = int(cfg.get("seq_len", seq_len))
            metrics = evaluate_gpt_on_val(
                model=model,
                tokens=tokens,
                seq_len=seq_len,
                device=device,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"[benchmark] Loaded metrics for {name} from {json_path}")

        project_metrics.append((name, metrics))

        if not args.skip_generation:
            print(f"[benchmark] Generating samples for {name}...")
            samples = generate_project_samples(
                ckpt_path, prompts, max_new_tokens, temperature, top_k, device
            )
            project_samples.append((name, samples))

    hf_metrics: list[tuple[str, dict]] = []
    hf_samples: list[tuple[str, list[tuple[str, str]]]] = []

    for model_name in gpt2_models:
        print(f"[benchmark] Evaluating HF model: {model_name}...")
        metrics = evaluate_hf_gpt2_on_val(
            model_name=model_name,
            tokens=tokens,
            seq_len=seq_len,
            device=device,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        hf_metrics.append((model_name, metrics))
        print(
            f"[benchmark] {model_name}: loss={metrics['val_loss']:.4f} "
            f"ppl={metrics['perplexity']:.2f}"
        )

        if not args.skip_generation:
            print(f"[benchmark] Generating samples for {model_name}...")
            samples = generate_hf_samples(
                model_name, prompts, max_new_tokens, temperature, top_k, device
            )
            hf_samples.append((model_name, samples))

    write_report(
        report_path,
        footnote,
        project_metrics,
        hf_metrics,
        project_samples,
        hf_samples,
    )


if __name__ == "__main__":
    main()
