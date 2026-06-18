"""
data_utils.py — Streaming dataset for TinyStories text files.

WHY streaming (IterableDataset) instead of loading everything into RAM?
  TinyStories train split is ~2 GB of raw text. Tokenised, that's hundreds of
  millions of integers. Loading it all into RAM would either crash the job or
  waste memory that the GPU needs. Instead we stream: read the file line by
  line, tokenise on the fly, and slice out fixed-length chunks.

HOW a language model training batch works:
  The model learns to predict the next token. Given a sequence of tokens:
    [t0, t1, t2, ..., t_{L-1}]
  The input (x) is every token except the last:
    x = [t0, t1, ..., t_{L-2}]
  The target (y) is every token except the first, shifted one position right:
    y = [t1, t2, ..., t_{L-1}]
  So at position i, the model sees tokens 0..i and must predict token i+1.
  This gives us L-1 prediction tasks from a single sequence of length L.

  With seq_len=1024 we get 1023 predictions per sequence, which is very
  efficient compared to predicting only one token at a time.
"""

import os
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader

from src.tokenizer import Tokenizer


class TinyStoriesDataset(IterableDataset):
    """
    Streams tokenised chunks from a TinyStories .txt file.

    The file format is stories separated by the <|endoftext|> sentinel:
        Once upon a time...<|endoftext|>
        In a small village...<|endoftext|>
        ...

    We read the file line-by-line, tokenise each line, and accumulate tokens
    into a rolling buffer. Whenever the buffer has at least (seq_len + 1)
    tokens we yield one training chunk and advance the buffer.

    The +1 is because we need seq_len tokens for x AND seq_len tokens for y,
    and x[i+1] == y[i], so a chunk of length seq_len+1 gives us both.
    """

    def __init__(self, file_path: str, seq_len: int) -> None:
        """
        Args:
            file_path: Absolute path to the .txt file on $SCRATCH.
            seq_len:   Number of tokens per training example (e.g. 1024).
        """
        super().__init__()

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset not found: {file_path}\n"
                "Download with: wget https://huggingface.co/datasets/roneneldan/"
                "TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
            )

        self.file_path = file_path
        self.seq_len = seq_len
        self.tokenizer = Tokenizer()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yields (x, y) pairs of shape (seq_len,) each.

        x: input tokens — indices 0 .. seq_len-1
        y: target tokens — indices 1 .. seq_len   (shifted by one)

        Both are int64 (Long) tensors, which is what nn.Embedding expects.
        """
        # A rolling list of integer token IDs accumulated across lines.
        # We fill it up and drain seq_len+1 chunks from the front.
        buffer: list[int] = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Tokenise the line. Lines ending with <|endoftext|> get the
                # special EOT token appended, signalling a story boundary.
                tokens = self.tokenizer.encode(line, add_eot=False)
                buffer.extend(tokens)

                # Drain complete chunks from the buffer.
                # We keep looping as long as we have enough tokens for one chunk.
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len + 1 :]  # advance past consumed tokens

                    # x is the input — everything except the last token.
                    # y is the target — everything except the first token.
                    # At position i: model sees x[0..i], must predict y[i] = x[i+1].
                    x = torch.tensor(chunk[: self.seq_len], dtype=torch.long)
                    y = torch.tensor(chunk[1 : self.seq_len + 1], dtype=torch.long)
                    yield x, y


def build_dataloader(
    file_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    """
    Wrap TinyStoriesDataset in a DataLoader for batched training.

    WHY num_workers > 0?
      The DataLoader spawns worker processes that prefetch batches while the
      GPU is busy with the forward/backward pass. Without this the GPU would
      sit idle waiting for CPU tokenisation. num_workers=2 is a safe default
      for a single-GPU job with 4 CPUs allocated.

    Args:
        file_path:   Path to the .txt data file.
        seq_len:     Token sequence length (e.g. 1024).
        batch_size:  Number of sequences per batch (e.g. 64).
        num_workers: CPU worker processes for prefetching.

    Returns:
        A DataLoader that yields (x, y) tensors of shape (batch_size, seq_len).
    """
    dataset = TinyStoriesDataset(file_path=file_path, seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # pin_memory=True copies batches to page-locked RAM, making the
        # CPU→GPU transfer faster (uses DMA instead of going through the OS).
        pin_memory=True,
        num_workers=num_workers,
        # persistent_workers=True keeps worker processes alive between epochs
        # so we don't pay the process-spawn cost on every iteration.
        persistent_workers=(num_workers > 0),
    )
