"""
tokenizer.py — Wrapper around tiktoken for GPT-2 byte-pair encoding.

WHY tiktoken?
  tiktoken is OpenAI's fast BPE tokenizer. We use the "gpt2" encoding, which
  has a fixed vocabulary of 50,257 tokens. Every word is broken into subword
  pieces so the model never sees truly "unknown" tokens — it just sees rarer
  subword combinations.

  Example:  "hello world" → [31373, 995]
            "unrecognized" → [403, 8608, 1143]   (split into subwords)

HOW BPE works (conceptually):
  1. Start with individual characters as the vocabulary.
  2. Count the most frequent adjacent pair of tokens in the corpus.
  3. Merge that pair into a single new token.
  4. Repeat until the vocabulary reaches the target size (50,257 for GPT-2).
  The result: common words become a single token; rare words are split.
"""

import tiktoken


# The GPT-2 vocabulary has exactly 50,257 tokens.
# This is a fixed constant — the model's embedding table will have this many rows.
VOCAB_SIZE: int = 50257

# tiktoken uses a special "end of text" token (<|endoftext|>) to mark
# the boundary between documents. Its integer ID in the GPT-2 vocab is 50256.
# We use this to separate stories in TinyStories and to signal generation stop.
EOT_TOKEN_ID: int = 50256


class Tokenizer:
    """
    Thin wrapper around tiktoken's GPT-2 encoding.

    Why wrap at all? Two reasons:
      1. Centralise the encoding choice — if we ever switch tokenisers,
         only this file changes.
      2. Expose encode/decode with consistent type signatures for the rest
         of the codebase.
    """

    def __init__(self) -> None:
        # Load the GPT-2 BPE encoding from tiktoken.
        # This downloads a small vocab file on first use, then caches it.
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        """
        Convert a string into a list of integer token IDs.

        Args:
            text:    The raw string to tokenise.
            add_eot: If True, append the end-of-text token (50256) after the
                     tokens. Useful when encoding individual documents so the
                     model learns where documents end.

        Returns:
            A list of integer token IDs.

        Example:
            tokenizer.encode("Once upon a time")
            → [7454, 2402, 257, 640]
        """
        # allowed_special tells tiktoken which special tokens to recognise
        # in the raw text. "all" means <|endoftext|> in the source text is
        # turned into token 50256 rather than being encoded character-by-character.
        ids = self._enc.encode(text, allowed_special={"<|endoftext|>"})
        if add_eot:
            ids.append(EOT_TOKEN_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token IDs back into a human-readable string.

        Note: decode is lossy for bytes that don't form valid UTF-8 on their
        own (tiktoken handles this gracefully with error replacement).
        """
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Always 50,257 for GPT-2 encoding."""
        return VOCAB_SIZE

    @property
    def eot_token_id(self) -> int:
        """Integer ID of the <|endoftext|> special token."""
        return EOT_TOKEN_ID
