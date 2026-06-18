"""
model.py — GPT-2 style decoder-only transformer from scratch.

ARCHITECTURE OVERVIEW
─────────────────────
A GPT is a stack of identical "transformer blocks". Data flows like this:

  Token IDs (integers)
       │
       ▼
  Embedding Layer  — maps each token ID to a dense vector of size n_embd
       │            (also adds positional information)
       ▼
  TransformerBlock × n_layer
  ┌──────────────────────────────────────┐
  │  LayerNorm                           │
  │       │                              │
  │  CausalSelfAttention  ← masked,      │
  │       │                 so each      │
  │  residual add           position     │
  │       │                 only sees    │
  │  LayerNorm              the past     │
  │       │                              │
  │  MLP (2-layer feedforward)           │
  │       │                              │
  │  residual add                        │
  └──────────────────────────────────────┘
       │
       ▼
  Final LayerNorm
       │
       ▼
  LM Head (Linear)  — projects n_embd → vocab_size logits
       │
       ▼
  Logits: probability distribution over next token

KEY CONCEPTS
────────────
- Residual connections: each block ADDS its output to its input. This lets
  gradients flow directly back to early layers without vanishing.

- Pre-norm (LayerNorm before attention/MLP): more stable than the original
  post-norm used in "Attention is All You Need". GPT-2 uses pre-norm.

- Causal masking: ensures position i can only attend to positions 0..i.
  Without this the model could "cheat" by looking at future tokens.

- Weight tying: the token embedding matrix and the LM head weight matrix
  are shared. This halves the parameter count for those layers and often
  improves perplexity.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """
    All model hyperparameters in one place.
    These are loaded from configs/phase1_124M.yaml via OmegaConf in the
    training script and passed here as a dataclass for type safety.
    """
    vocab_size: int = 50257   # Fixed: GPT-2 tiktoken vocabulary
    seq_len: int = 1024       # Maximum context length (how far back the model looks)
    n_layer: int = 12         # Number of stacked transformer blocks
    n_head: int = 12          # Number of attention heads per block
    n_embd: int = 768         # Embedding dimension (must be divisible by n_head)
    dropout: float = 0.1      # Dropout probability (0 = off, used during training)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "GPTConfig":
        """Build a GPTConfig from a merged OmegaConf DictConfig."""
        return cls(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        )


# ── Causal Self-Attention ───────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    WHAT IS ATTENTION?
    ─────────────────
    Each token asks: "which other tokens should I pay attention to?"

    For each token, we compute three vectors:
      Q (Query)  — "what am I looking for?"
      K (Key)    — "what do I contain?"
      V (Value)  — "what information do I give if attended to?"

    Attention score from token i to token j:
      score(i, j) = Q_i · K_j / sqrt(head_dim)

    The sqrt(head_dim) scaling prevents dot-products from getting too large
    (which would push softmax into regions with vanishingly small gradients).

    After softmax over scores, we get attention weights (sum to 1 over j).
    The output for token i is the weighted sum of all V_j values.

    MULTI-HEAD:
    ──────────
    Instead of one set of Q/K/V, we have n_head independent sets ("heads"),
    each operating on a n_embd/n_head dimensional subspace.
    Different heads learn to attend to different kinds of relationships
    (e.g. syntax in one head, coreference in another).
    Outputs are concatenated and projected back to n_embd.

    CAUSAL MASK:
    ───────────
    We set score(i, j) = -inf for all j > i before softmax.
    After softmax, those positions get weight ≈ 0, so token i cannot
    attend to any future token j. This makes the model autoregressive:
    it can only use past context to predict the next token.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Dimension of each attention head's Q/K/V vectors.
        self.head_dim = config.n_embd // config.n_head

        # Single linear layer that produces Q, K, V for ALL heads at once.
        # Output size is 3 * n_embd: first n_embd = Q, next = K, last = V.
        # Using one fused projection is more efficient than three separate ones.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection: after concatenating all heads, project back to n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Dropout applied to attention weights (randomly zeros some connections).
        self.attn_dropout = nn.Dropout(config.dropout)
        # Dropout applied to the output projection.
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: a lower-triangular matrix of ones.
        # Shape: (1, 1, seq_len, seq_len) — the extra dims allow broadcasting
        # over batch and head dimensions.
        # Registered as a buffer (not a parameter): saved with model, not trained.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
                1, 1, config.seq_len, config.seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
        Returns:
            (batch, seq_len, n_embd)
        """
        B, T, C = x.shape  # batch, time (sequence length), channels (n_embd)

        # ── Step 1: compute Q, K, V for all heads in one matrix multiply ──────
        # qkv shape: (B, T, 3*C)
        qkv = self.c_attn(x)
        # Split the last dimension into three equal parts.
        q, k, v = qkv.split(self.n_embd, dim=2)

        # ── Step 2: reshape for multi-head attention ───────────────────────────
        # We want shape: (B, n_head, T, head_dim)
        # .view splits C into (n_head, head_dim), .transpose moves n_head before T.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── Step 3: scaled dot-product attention ──────────────────────────────
        # scores shape: (B, n_head, T, T)
        # Each [b, h, i, j] is the attention score from token i to token j,
        # for batch b, head h.
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask: positions where mask==0 are set to -inf.
        # After softmax, -inf → 0, so those positions contribute nothing.
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax converts raw scores into weights that sum to 1 across the T dimension.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (B, n_head, T, head_dim)
        out = attn_weights @ v

        # ── Step 4: concatenate heads and project ─────────────────────────────
        # Transpose back: (B, T, n_head, head_dim), then flatten heads → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection mixes information across heads.
        return self.resid_dropout(self.c_proj(out))


# ── MLP (Feed-Forward Network) ──────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Position-wise feed-forward network applied after attention.

    WHY does a transformer need an MLP after attention?
    ──────────────────────────────────────────────────
    Attention lets tokens communicate with each other (mixing information
    across positions). The MLP then processes each token independently,
    giving the model capacity to transform representations in place.

    GPT-2 uses a 4x expansion: the hidden layer is 4 * n_embd wide.
    This expansion-compression structure lets the network learn a richer
    set of features before projecting back down.

    GELU activation:
    ───────────────
    GELU (Gaussian Error Linear Unit) is a smooth approximation to ReLU.
    It has a small negative region for slightly negative inputs (unlike ReLU
    which is exactly 0). This slight "leakiness" helps gradient flow.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        # Expand from n_embd to 4*n_embd (the "widening" step).
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.act = nn.GELU()
        # Project back from 4*n_embd to n_embd (the "narrowing" step).
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embd) → (B, T, 4*n_embd) → (B, T, n_embd)
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ── Transformer Block ───────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One transformer block = LayerNorm + Attention + residual
                          + LayerNorm + MLP + residual.

    PRE-NORM vs POST-NORM:
    ─────────────────────
    Original "Attention is All You Need" used post-norm (normalise after
    adding the residual). GPT-2 uses pre-norm: normalise the input BEFORE
    attention/MLP, then add to the unnormalised residual.

    Pre-norm is more training-stable, especially at depth, because the
    residual stream is never normalised — gradients flow through it cleanly.

    RESIDUAL CONNECTIONS (why they matter):
    ───────────────────────────────────────
    Without residuals, signal must pass through all 12 layers sequentially.
    Gradients would shrink exponentially as they backpropagate (vanishing
    gradient problem). With residuals, gradients can bypass any block and
    flow directly to earlier layers: dL/d(input) = dL/d(output) + ...
    This makes training deep networks feasible.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # LayerNorm normalises across the n_embd dimension for each token.
        # It stabilises activations and speeds up training.
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection.
        # x + attn(...): the original x is added back, preserving the signal.
        x = x + self.attn(self.ln1(x))
        # MLP sub-layer with residual connection.
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full GPT Model ──────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    GPT-2 style decoder-only transformer.

    TOKEN + POSITIONAL EMBEDDINGS:
    ──────────────────────────────
    Transformers have no inherent sense of order — attention treats all
    positions equally. To inject order, we add a learned positional embedding
    to each token embedding. The model learns to use position information
    through training.

      final_embedding[i] = token_embedding[token_id[i]] + pos_embedding[i]

    LM HEAD + WEIGHT TYING:
    ───────────────────────
    The LM head projects the final hidden state (n_embd) to logits over the
    vocabulary (vocab_size). We tie its weights to the token embedding matrix:
      lm_head.weight = token_embedding.weight

    Intuition: the embedding maps token IDs → vectors; the LM head maps
    vectors → token IDs. Sharing weights enforces a consistent representation
    of each token in both roles, and reduces parameters by ~38M for 124M GPT-2.

    PARAMETER COUNT (124M):
    ───────────────────────
      Token embeddings:    50257 × 768 ≈  38.6M  (shared with lm_head)
      Positional embeddings: 1024 × 768 ≈   0.8M
      Per block: attn (~2.4M) + mlp (~4.7M) + norms ≈  7.1M × 12 layers ≈ 85.2M
      Final LayerNorm + LM head weight (tied): ≈ 0.1M
      Total: ≈ 124M parameters
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding: maps each token ID (integer) to a dense vector.
        # Shape of weight matrix: (vocab_size, n_embd)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional embedding: a separate learned vector for each position 0..seq_len-1.
        # Shape: (seq_len, n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # The core: a sequence of identical transformer blocks.
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Final layer norm applied after all transformer blocks.
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM head: projects hidden states to vocabulary logits.
        # bias=False is standard for GPT-2.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: make lm_head use the same weight matrix as tok_emb.
        # After this line, lm_head.weight IS tok_emb.weight (same tensor).
        self.lm_head.weight = self.tok_emb.weight

        # Initialise weights using GPT-2's scheme.
        self.apply(self._init_weights)

        # Special scaling for residual projections: divide by sqrt(2 * n_layer).
        # WHY? Each block adds its output to the residual stream. With n_layer=12
        # blocks, the residual stream grows. Scaling the output projections down
        # keeps the variance of activations stable regardless of depth.
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 weight initialisation:
          - Linear layers: normal distribution with std=0.02
          - Embeddings: same
          - LayerNorm: weight=1, bias=0 (identity transform at init)

        std=0.02 is small enough to keep activations from exploding at init.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            idx:     Token indices, shape (B, T). Values in [0, vocab_size).
            targets: Target token indices, shape (B, T). If provided, also
                     compute and return cross-entropy loss.

        Returns:
            (logits, loss) where:
              logits: (B, T, vocab_size) — raw scores over next token
              loss:   scalar cross-entropy loss, or None if targets not given
        """
        B, T = idx.shape
        assert T <= self.config.seq_len, (
            f"Sequence length {T} exceeds model context length {self.config.seq_len}"
        )

        # ── Embeddings ────────────────────────────────────────────────────────
        # Create position indices [0, 1, 2, ..., T-1] for the current sequence.
        positions = torch.arange(T, device=idx.device)  # shape: (T,)

        # Look up token and position embeddings and sum them.
        # Broadcasting: tok_emb is (B, T, n_embd), pos_emb is (T, n_embd).
        # PyTorch broadcasts pos_emb to match the batch dimension automatically.
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        # ── Transformer blocks ────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)

        # ── Final layer norm + LM head ────────────────────────────────────────
        x = self.ln_f(x)

        # Project to vocabulary: (B, T, n_embd) → (B, T, vocab_size)
        logits = self.lm_head(x)

        # ── Loss (optional) ───────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # cross_entropy expects (N, C) logits and (N,) targets.
            # We flatten the batch and time dimensions: (B*T, vocab_size) and (B*T,).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens, appending each to the context.

        TEMPERATURE: dividing logits by temperature controls randomness.
          temperature < 1.0 → sharper distribution → more conservative text
          temperature > 1.0 → flatter distribution → more creative/random text
          temperature = 1.0 → unmodified model distribution

        TOP-K SAMPLING: instead of sampling from all 50,257 tokens, restrict
        to the top-k most probable tokens and renormalise. This prevents the
        model from ever sampling very unlikely tokens, which often produce
        incoherent text.

        Args:
            idx:            Seed token indices, shape (B, T).
            max_new_tokens: How many tokens to generate.
            temperature:    Sampling temperature (default 1.0).
            top_k:          If set, restrict sampling to top-k tokens.

        Returns:
            Token indices including the generated tokens, shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop context to the model's maximum sequence length.
            # If the running sequence is longer than seq_len, take only the last seq_len tokens.
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len :]

            # Forward pass — only logits needed for generation (no targets).
            logits, _ = self(idx_cond)

            # Take the logits at the last position: these are scores for the next token.
            logits = logits[:, -1, :] / temperature  # shape: (B, vocab_size)

            # Optional top-k filtering.
            if top_k is not None:
                # Find the k-th largest logit value.
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits below the k-th value to -inf so they get ~0 probability.
                logits[logits < top_values[:, [-1]]] = float("-inf")

            # Convert logits to probabilities via softmax.
            probs = F.softmax(logits, dim=-1)

            # Sample one token from the distribution.
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            # Append the new token to the running context.
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
