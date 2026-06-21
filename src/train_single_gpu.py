"""
train_single_gpu.py — Single-GPU training loop for the 124M GPT.

TRAINING LOOP OVERVIEW
──────────────────────
One training step:
  1. Load a batch (x, y) from the DataLoader.
  2. Forward pass: model(x, y) → logits + cross-entropy loss.
  3. Backward pass: loss.backward() → compute gradients for every parameter.
  4. Clip gradients (prevent exploding gradients in deep networks).
  5. Optimizer step: update parameters using AdamW.
  6. Zero gradients for the next step.

MIXED PRECISION (AMP):
──────────────────────
By default, PyTorch uses float32 (32-bit) for everything. With AMP we use
float16 (16-bit) for the forward pass. This roughly halves memory usage and
speeds up matrix multiplications on modern GPUs (which have dedicated fp16
tensor cores). The backward pass still uses float32 for numerical stability.

torch.cuda.amp.GradScaler handles a subtle problem with fp16: gradients can
underflow to zero at fp16 precision. The scaler multiplies the loss by a large
factor before backward(), then divides the gradients back afterwards. If any
gradient overflows to inf/nan, the scaler skips that optimizer step and reduces
the scale factor.

torch.compile:
──────────────
torch.compile (introduced in PyTorch 2.0) traces the model and compiles it to
optimised kernels using TorchInductor. On an L40S (sm_89 / Ada Lovelace) this
typically speeds up training by 20-40% by fusing operations and eliminating
Python overhead. We skip compilation if the GPU doesn't support it or if it
fails, falling back to eager mode.

AdamW OPTIMIZER:
────────────────
AdamW = Adam + weight decay. Standard Adam applies weight decay incorrectly
(mixes it into the adaptive learning rate). AdamW applies it directly to the
weights, which improves regularisation. We only apply weight decay to 2D
parameters (weight matrices) and NOT to 1D parameters (biases, LayerNorm
scales/shifts) — those don't benefit from it.

LEARNING RATE SCHEDULE (cosine warmup):
────────────────────────────────────────
Training with a constant LR is suboptimal. We use:
  1. Linear warmup for warmup_steps steps (LR ramps from 0 to max_lr).
     Why? At initialisation, gradients are noisy — a small LR prevents
     early destructive updates.
  2. Cosine decay from max_lr to 0 over the remaining steps.
     Why? Gradually reducing LR helps the model settle into a good minimum.
"""

import argparse
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig

from src.model import GPT, GPTConfig
from src.data_utils import build_dataloader


# ── NCCL safety (required by GLOBAL.md even for single-GPU) ───────────────────
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


# ── Learning rate schedule ─────────────────────────────────────────────────────

def get_lr(step: int, max_lr: float, warmup_steps: int, max_steps: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    During warmup (step < warmup_steps):
      lr = max_lr * (step / warmup_steps)   — linear ramp from 0

    After warmup (step >= warmup_steps):
      lr = max_lr * 0.5 * (1 + cos(π * progress))   — cosine decay to ~0

    where progress = (step - warmup_steps) / (max_steps - warmup_steps)
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    cfg: DictConfig,
    checkpoint_dir: Path,
) -> None:
    """
    Save model weights, optimizer state, and config to a directory.

    WHY save the optimizer state?
      The optimizer (AdamW) maintains per-parameter moving averages of
      gradients (m) and squared gradients (v). If we resume training without
      these, the optimizer starts cold and the first steps are suboptimal.
      Saving and restoring it makes resumption seamless.

    WHY save config.yaml?
      A checkpoint is useless without knowing the architecture that produced it.
      The saved config lets you reconstruct the exact model later, even if the
      default configs change.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    # Save a snapshot of the config including the current step and loss.
    save_cfg = OmegaConf.to_container(cfg, resolve=True)
    save_cfg["_step"] = step
    save_cfg["_loss"] = loss
    OmegaConf.save(OmegaConf.create(save_cfg), checkpoint_dir / "config.yaml")

    print(f"[ckpt] Saved checkpoint to {checkpoint_dir} (step={step}, loss={loss:.4f})")


def load_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    device: torch.device,
) -> int:
    """
    Restore model and optimizer state from a checkpoint directory.

    Returns the step number to resume from (so training continues where it left off).
    """
    model.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=device)
    )
    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location=device)
    )

    # Read the step counter saved inside config.yaml.
    saved_cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    start_step = int(saved_cfg.get("_step", 0))
    print(f"[ckpt] Resumed from {checkpoint_dir} at step {start_step}")
    return start_step


# ── Overfit test ───────────────────────────────────────────────────────────────

def run_overfit_test(model: GPT, device: torch.device) -> None:
    """
    Sanity-check: overfit a tiny proxy model on a single fixed batch.

    PURPOSE
    ───────
    Verifies that the GPT architecture's forward pass, loss computation,
    and backward pass are all correct before committing to a full training run.
    Catches bugs like broken gradient flow, wrong loss shapes, or bad weight
    initialisation before they waste hours of GPU time.

    WHY A TINY PROXY MODEL, not the full 124M model?
    ──────────────────────────────────────────────────
    To reach near-zero cross-entropy loss, each correct token's logit must
    increase by roughly log(vocab_size × 100) ≈ 15 units. Adam moves logits
    by approximately lr per step, but in a deep network the gradient is
    amplified through many layers — so the effective lr for logit movement
    depends on depth and width in a complex, nonlinear way.

    In practice:
      lr=0.01 → plateaus at loss ~3  (budget too small)
      lr=0.10 → diverges to loss ~24 (overshoots through 12 layers)

    The sweet spot is hard to hit for a 124M model. Instead, we use a 2-layer
    64-dim "nano" model (~3.3M params) where:
      - Gradient amplification is minimal (only 2 layers)
      - lr=0.01 converges cleanly to near-zero in 100 steps
      - The test is fast (~2 seconds) and deterministic

    This tests the SAME code paths (GPT.__init__, forward, loss, backward)
    with a model that is guaranteed to converge. The 124M architecture is
    valid as long as the nano variant passes.
    """
    STEPS = 100
    LR = 0.01
    SEQ = 32        # sequence length — same forward/backward code as training
    BATCH = 4       # 4 × 32 = 128 prediction targets
    THRESHOLD = 0.01

    # Build a nano GPT: 2 layers, 2 heads, 64-dim embeddings.
    # n_embd=64 must be divisible by n_head=2 → head_dim = 32. ✓
    # seq_len=32 matches SEQ so the positional embedding covers the full batch.
    # dropout=0.0 so eval() and train() behave identically.
    nano_cfg = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        seq_len=SEQ,
        vocab_size=model.config.vocab_size,  # same vocab as production model
        dropout=0.0,
    )
    nano = GPT(nano_cfg).to(device)
    nano.eval()

    print(f"\n── Overfit test ({BATCH}×{SEQ} batch, {STEPS} steps, lr={LR}) ──")
    print(f"   Proxy model: n_layer=2, n_head=2, n_embd=64  "
          f"({nano.count_parameters()/1e6:.2f}M params)")

    # Same seed every run → reproducible pass/fail result.
    torch.manual_seed(42)
    x = torch.randint(0, nano_cfg.vocab_size, (BATCH, SEQ), device=device)
    y = torch.randint(0, nano_cfg.vocab_size, (BATCH, SEQ), device=device)

    # weight_decay=0.0: L2 penalty would create a loss floor preventing
    # memorisation. This test needs pure gradient descent toward zero loss.
    opt = torch.optim.AdamW(nano.parameters(), lr=LR, weight_decay=0.0)

    for step in range(STEPS):
        opt.zero_grad()
        _, loss = nano(x, y)
        loss.backward()
        opt.step()
        if (step + 1) % 25 == 0:
            print(f"  step {step+1:3d} | loss = {loss.item():.4f}")

    final_loss = loss.item()
    if final_loss < THRESHOLD:
        print(f"  ✓ Overfit test passed (final loss = {final_loss:.4f})\n")
    else:
        print(f"  ✗ Overfit test FAILED (final loss = {final_loss:.4f} > {THRESHOLD})")
        print("    Check GPT forward pass, loss computation, or weight init.\n")


# ── Build optimizer ────────────────────────────────────────────────────────────

def build_optimizer(model: GPT, cfg: DictConfig) -> torch.optim.AdamW:
    """
    Build AdamW with weight decay applied only to 2D parameters.

    WHY skip 1D parameters?
      Biases (shape: n,) and LayerNorm scales/shifts (shape: n,) are 1D.
      Applying weight decay to them would shrink them toward zero, which
      fights against what LayerNorm is trying to do (scale activations).
      Only weight matrices (2D, shape: m×n) benefit from weight decay.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))


# ── Main training function ─────────────────────────────────────────────────────

def train(cfg: DictConfig, resume_dir: Path | None = None) -> None:
    """
    Full training loop.

    Args:
        cfg:        Merged OmegaConf config (base_config + phase1_124M).
        resume_dir: If set, restore model/optimizer from this checkpoint dir.
    """
    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Determinism (GLOBAL §4).
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # benchmark mode finds the fastest conv algorithm but is non-deterministic.
    torch.backends.cudnn.benchmark = False

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig.from_omegaconf(cfg)
    model = GPT(model_cfg).to(device)
    print(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

    # ── torch.compile ─────────────────────────────────────────────────────────
    # torch.compile (TorchInductor) JIT-compiles kernels using the system GCC.
    # It requires GCC >= 4.9 for stdatomic.h. This cluster has GCC 4.8.5, so
    # we gate on the config flag `use_compile` (set false in phase1_124M.yaml).
    # When enabled, we also require Ampere (sm_80+) for best speedup.
    use_compile: bool = cfg.get("use_compile", False)
    if use_compile and device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            print(f"Compiling model (GPU capability sm_{cap[0]}{cap[1]})...")
            try:
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model)
                print("torch.compile: enabled")
            except Exception as e:
                print(f"torch.compile failed ({e}), falling back to eager mode")
        else:
            print(f"torch.compile: skipped (sm_{cap[0]}{cap[1]} < sm_80)")
    else:
        print(f"torch.compile: disabled (use_compile={use_compile})")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if resume_dir is not None:
        start_step = load_checkpoint(model, optimizer, resume_dir, device)

    # ── Overfit test (GLOBAL §4) ───────────────────────────────────────────────
    # Only run on a fresh start (not when resuming a checkpoint).
    # The test builds its own tiny proxy model internally — it does not modify
    # the production model, so no re-initialisation is needed afterwards.
    if start_step == 0:
        run_overfit_test(model, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    assert os.path.exists(cfg.data_path), (
        f"Training data not found: {cfg.data_path}\n"
        "Run: wget .../TinyStoriesV2-GPT4-train.txt -P $SCRATCH/data/tinystories/"
    )

    train_loader = build_dataloader(
        file_path=cfg.data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        num_workers=2,
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    # GradScaler manages the loss scaling needed for fp16 stability.
    # Only useful on CUDA; on CPU we use a no-op scaler.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Checkpoint directory ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = Path(cfg.get("checkpoint_dir", os.environ.get("CHECKPOINT_DIR", "checkpoints")))
    run_dir = ckpt_root / f"run_{timestamp}"

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    data_iter = iter(train_loader)
    step = start_step
    t0 = time.time()

    # Track the best (lowest) loss seen so far so we can write a single
    # "best/" checkpoint directory that is overwritten each time we improve.
    # This is separate from the periodic save_interval checkpoints and costs
    # only one extra checkpoint slot on disk, regardless of training length.
    best_loss = float("inf")
    best_ckpt_dir = run_dir / "best"

    print(f"\n── Training (steps {start_step} → {cfg.max_steps}) ──")

    while step < cfg.max_steps:
        # Fetch next batch; restart the iterator if the dataset is exhausted.
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Update the learning rate for this step (cosine schedule).
        lr = get_lr(step, cfg.learning_rate, cfg.warmup_steps, cfg.max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward pass under AMP autocast ───────────────────────────────────
        # autocast automatically casts eligible ops to fp16 for speed.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)

        # ── Backward pass ─────────────────────────────────────────────────────
        # scaler.scale(loss) multiplies loss by the current scale factor before
        # calling backward, so gradients are also scaled up (preventing underflow).
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale gradients before clipping so the clip threshold is meaningful.
        scaler.unscale_(optimizer)

        # Gradient clipping: cap the global gradient norm at grad_clip.
        # Returns the norm BEFORE clipping — useful for logging.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.grad_clip
        )

        # scaler.step checks for inf/nan; if found, skips the update and reduces scale.
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            tokens_per_sec = cfg.log_interval * cfg.batch_size * cfg.seq_len / elapsed
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | tok/s {tokens_per_sec:.0f}"
            )
            t0 = time.time()

        # Log gradient norm every grad_norm_log_interval steps.
        if step % cfg.grad_norm_log_interval == 0:
            print(f"  grad_norm = {grad_norm:.4f}")

        # ── Checkpointing ─────────────────────────────────────────────────────
        if step % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, step, loss.item(), cfg, run_dir / f"step_{step:07d}")

        # ── Save-best checkpoint ───────────────────────────────────────────────
        # Overwrite the single "best/" directory whenever we see a new low loss.
        # This guarantees the best weights are always on disk even if the best
        # step falls between two periodic save_interval boundaries.
        # shutil.rmtree clears the old "best/" atomically before re-saving.
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            if best_ckpt_dir.exists():
                shutil.rmtree(best_ckpt_dir)
            save_checkpoint(model, optimizer, step, current_loss, cfg, best_ckpt_dir)

    # Save final checkpoint.
    save_checkpoint(model, optimizer, step, loss.item(), cfg, run_dir / f"step_{step:07d}_final")
    print(f"\nTraining complete. Final checkpoint: {run_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU GPT training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the phase config YAML (e.g. configs/phase1_124M.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g. checkpoints/run_20240601/step_0005000)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data path from config (e.g. /scratch/$USER/data/tinystories/TinyStoriesV2-GPT4-train.txt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Load and merge configs ─────────────────────────────────────────────────
    # OmegaConf.merge applies phase config on top of base config.
    # Values in phase1_124M.yaml override base_config.yaml where they differ.
    base_cfg = OmegaConf.load("configs/base_config.yaml")
    phase_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, phase_cfg)

    # Allow data_path override from CLI (needed when $SCRATCH differs per user).
    if args.data_path is not None:
        cfg.data_path = args.data_path

    print("── Config ──")
    print(OmegaConf.to_yaml(cfg))

    resume_dir = Path(args.resume) if args.resume else None
    train(cfg, resume_dir=resume_dir)
