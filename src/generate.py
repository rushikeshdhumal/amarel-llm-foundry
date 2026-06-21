"""
generate.py — Load a trained GPT checkpoint and generate text.

HOW AUTOREGRESSIVE GENERATION WORKS
─────────────────────────────────────
A GPT is trained to predict the NEXT token given all previous tokens.
At inference time we exploit this by running a loop:

  1. Feed the prompt tokens through the model → get logits over the vocabulary
     for the LAST position (what comes next?).
  2. Convert logits to probabilities with softmax.
  3. Sample one token from that distribution.
  4. Append the new token to the context and repeat.

The model never sees the future — it only ever predicts one token ahead —
which is why this is called "autoregressive" (the output feeds back as input).

TEMPERATURE
───────────
Before softmax we divide logits by a temperature T:
  T < 1.0 → sharper distribution → model picks high-probability tokens more
            consistently → safer, repetitive text
  T = 1.0 → raw model distribution (default)
  T > 1.0 → flatter distribution → more creative/varied but can be incoherent

For a model trained on TinyStories a temperature of 0.8–1.0 works well.

TOP-K SAMPLING
──────────────
Instead of sampling from all 50,257 tokens, we keep only the top-k most
probable tokens and renormalise their probabilities to sum to 1.
  top_k=1  → deterministic greedy decoding (always pick the most likely token)
  top_k=40 → standard GPT-2 sampling — restricts to plausible tokens while
             still allowing variety

CHECKPOINT LAYOUT
─────────────────
Each checkpoint directory (e.g. checkpoints/run_<ts>/step_0091500/) contains:
  model.pt       — model state dict (weights)
  optimizer.pt   — optimizer state (not needed for inference, only for resuming)
  config.yaml    — merged OmegaConf config with _step and _loss fields added

This script reads model.pt and config.yaml; it does NOT need optimizer.pt.
"""

import argparse
import sys
from pathlib import Path

import torch
import tiktoken
from omegaconf import OmegaConf

# Allow running as: python -m src.generate OR python src/generate.py from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import GPT, GPTConfig


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[GPT, dict]:
    """
    Reconstruct the GPT model from a checkpoint directory and load weights.

    WHY read config.yaml from the checkpoint?
      The checkpoint's config.yaml is a snapshot of the exact architecture
      used when that checkpoint was saved. If base_config.yaml or phase config
      later changes, we still reconstruct the correct model shape.

    Returns:
        (model, saved_cfg)  — model in eval mode, raw config dict for display.
    """
    config_path = checkpoint_dir / "config.yaml"
    weights_path = checkpoint_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {checkpoint_dir}.\n"
            "Make sure you point to a checkpoint directory, not a run directory."
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"No model.pt found in {checkpoint_dir}.")

    # Load the config snapshot from inside the checkpoint.
    cfg = OmegaConf.load(config_path)
    saved_step = cfg.get("_step", "?")
    saved_loss = cfg.get("_loss", None)

    # Build the model architecture from the saved config.
    model_cfg = GPTConfig.from_omegaconf(cfg)
    model = GPT(model_cfg).to(device)

    # Load weights.
    # map_location=device ensures tensors land on the right device even if
    # the checkpoint was saved on a different GPU index.
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # eval() disables dropout — essential for deterministic inference.
    # During training dropout randomly zeros activations; at inference we
    # want the full model signal without stochastic noise.
    model.eval()

    loss_str = f", loss={saved_loss:.4f}" if saved_loss is not None else ""
    print(f"[generate] Loaded checkpoint: {checkpoint_dir}")
    print(f"           step={saved_step}{loss_str}")
    print(f"           architecture: {model_cfg.n_layer}L / {model_cfg.n_head}H / "
          f"{model_cfg.n_embd}D  ({model.count_parameters()/1e6:.1f}M params)")

    return model, OmegaConf.to_container(cfg, resolve=True)


# ── Text generation ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    """
    Tokenize a text prompt, run autoregressive generation, decode the result.

    Args:
        model:          GPT model in eval mode.
        enc:            tiktoken GPT-2 encoder/decoder.
        prompt:         Text to condition generation on.
        max_new_tokens: Number of new tokens to generate (not counting the prompt).
        temperature:    Sampling temperature (see module docstring).
        top_k:          Top-k sampling cutoff; None = no cutoff (full distribution).
        device:         CUDA or CPU device.

    Returns:
        Generated text (does NOT include the prompt — just the completion).
    """
    # Encode prompt: list of integer token IDs.
    # allowed_special="all" lets tiktoken pass through special tokens like <|endoftext|>.
    prompt_tokens = enc.encode(prompt, allowed_special="all")
    n_prompt = len(prompt_tokens)

    # Build a (1, T) tensor — batch size 1 for single-sequence inference.
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # GPT.generate() runs the autoregressive loop and returns (1, T+max_new_tokens).
    out_idx = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)

    # Decode only the new tokens (slice off the prompt portion).
    new_tokens = out_idx[0, n_prompt:].tolist()
    return enc.decode(new_tokens)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained GPT checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using the nearest checkpoint to the best Phase 1 loss:
  python -m src.generate \\
    --checkpoint $SCRATCH/checkpoints/run_<ts>/step_0091500 \\
    --prompts "Once upon a time" "There was a little girl"

  # Using the saved-best checkpoint (Phase 2+ runs will have this):
  python -m src.generate \\
    --checkpoint $SCRATCH/checkpoints/run_<ts>/best \\
    --max_new_tokens 200 --temperature 0.9 --top_k 40
""",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (contains model.pt + config.yaml).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "Once upon a time there was a little girl named Lily",
            "One day, a small dog found a",
            "The sun was shining and",
        ],
        help="One or more text prompts to complete. Default: 3 TinyStories-style seeds.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Number of new tokens to generate per prompt (default: 150).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature: <1 = conservative, >1 = creative (default: 0.8).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling: restrict to k most-probable tokens (default: 40).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cuda', 'cpu'. Default: auto-detect CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")
    if device.type == "cuda":
        print(f"[generate] GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint)
    model, _ = load_model_from_checkpoint(checkpoint_dir, device)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # tiktoken's "gpt2" encoding matches the vocab the model was trained on.
    # It uses byte-pair encoding (BPE): common subwords get single token IDs,
    # rare/unknown text is split into smaller byte-level pieces.
    enc = tiktoken.get_encoding("gpt2")
    print(f"[generate] Tokenizer: gpt2 BPE  (vocab_size={enc.n_vocab})")

    # ── Generate ──────────────────────────────────────────────────────────────
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  max_new_tokens={args.max_new_tokens}  "
          f"temperature={args.temperature}  top_k={args.top_k}")
    print(sep)

    for i, prompt in enumerate(args.prompts, 1):
        print(f"\n── Prompt {i}/{len(args.prompts)} ──")
        print(f"  \033[1m{prompt}\033[0m")   # bold prompt in terminal
        print("  ↓ generated continuation:")

        completion = generate_text(
            model=model,
            enc=enc,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        # Print prompt + completion together for readability.
        full_text = prompt + completion
        print(f"  {full_text}")

    print(f"\n{sep}")
    print("  Generation complete.")
    print(sep)


if __name__ == "__main__":
    main()
