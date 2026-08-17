# Amarel LLM Foundry

A from-scratch GPT training stack on [Amarel](https://oarc.rutgers.edu/amarel/), Rutgers' HPC cluster. The model is a decoder-only transformer in pure PyTorch (`torch.nn` — no Hugging Face Trainer, no Lightning). Training scales from a single GPU through intra-node DDP to four-node FSDP; evaluation measures the resulting checkpoints against Hugging Face GPT-2 on a fixed TinyStories holdout.

| | 124M | 355M | 1.42B |
| :--- | :---: | :---: | :---: |
| Hardware | 1 GPU | 1 node × 4 GPU | 4 nodes × 4 GPU |
| Parallelism | — | DDP | FSDP |
| Architecture | 12L / 12H / 768D | 24L / 16H / 1024D | 24L / 16H / 2048D |

**Data:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (`TinyStoriesV2-GPT4`). **Tokenizer:** GPT-2 BPE via `tiktoken` (`vocab_size=50257`, `seq_len=1024`). **Cluster:** SLURM + NCCL.

Implementation lives on a dedicated branch per scaling stage. This file is the project overview; runnable code is on the feature branches linked below.

---

## Results

TinyStories validation, 1M tokens over 976 non-overlapping 1024-token windows. Lower is better.

| Model | Params | Val loss | Perplexity | Bits/token |
| :--- | ---: | ---: | ---: | ---: |
| Foundry 124M | 124M | 1.0848 | 2.96 | 1.5650 |
| Foundry 355M | 355M | 0.9460 | 2.58 | 1.3648 |
| **Foundry 1.42B** | **1.42B** | **0.9311** | **2.54** | **1.3433** |
| `gpt2` | 124M | 2.4708 | 11.83 | 3.5646 |
| `gpt2-medium` | 355M | 2.1944 | 8.97 | 3.1658 |
| `gpt2-large` | 774M | 2.0467 | 7.74 | 2.9528 |
| `gpt2-xl` | 1.5B | 1.9625 | 7.12 | 2.8313 |

*Foundry models: TinyStories pretrain. GPT-2: WebText. Same GPT-2 BPE tokenizer; 1.42B vs `gpt2-xl` is param-count approximate, not architecture-matched. Not matched on training tokens or corpus.*

The Foundry models beat every GPT-2 size on this holdout. That comparison is tokenizer- and architecture-fair, not training-data-fair: GPT-2 was trained on WebText, and TinyStories is a narrow children's-story distribution. Perplexity here measures in-domain fit, not general language modeling.

Two further observations:

- **Scaling flattens at 1.42B.** 4× the parameters relative to 355M bought 0.015 nats. The 1.42B run stopped at step 42,312 of 50,000 with an effective batch of 32 sequences (16 GPUs × 2). Token count is comparable to the smaller runs (~1.4B tokens); Chinchilla-optimal for this size would be an order of magnitude more. The model is undertrained, not architecturally limited.
- **Out-of-domain generation does not follow the table.** Given a business-news prompt, the 1.42B model reverts to a children's story while `gpt2-xl` stays in register:

> **Prompt:** In a surprising turn of events, the company announced

| Model | Completion |
| :--- | :--- |
| Foundry 1.42B | …, "We have a special gift for you, Sue!" Sue ran to the other side of the park and saw a huge pile of presents wrapped in colorful paper… |
| `gpt2-xl` | …on Monday that it would stop selling the Lumia 920 and the Lumia 820 on Friday, November 19. The company did not go into details of the reason for the discontinuation… |

In-domain completions ("Once upon a time there was a little girl named Lily") are coherent at every scale, with no obvious quality gap between 124M and 1.42B — the qualitative match to the flat perplexity move. Full samples for every prompt and model are on the [eval branch](https://github.com/rushikeshdhumal/amarel-llm-foundry/blob/feature/04-eval-closure/README.md).

---

## Architecture

All three models share the same GPT-2-style decoder-only stack:

```
Token IDs → tok_emb + pos_emb
         → × L  TransformerBlock (LN → causal MHA → residual → LN → MLP → residual)
         → LN → lm_head → next-token logits
```

Attention is multi-head causal self-attention; the MLP is a 4× expansion with GELU. Dropout is 0.1. The construction-time design ties `lm_head.weight` to `tok_emb.weight`; FSDP does not preserve that tie (see below), so the 1.42B run actually trains an extra embedding table (~103M parameters on top of the tied 1.31B count).

Training uses AdamW, gradient clipping at 1.0, linear warmup, AMP, and mixed-precision where applicable. Linear LR scaling with world size is applied under DDP and FSDP.

| | 124M | 355M | 1.42B |
| :--- | ---: | ---: | ---: |
| Per-GPU batch | 16 | 4 | 2 |
| World size | 1 | 4 | 16 |
| Effective batch (sequences) | 16 | 16 | 32 |
| Base LR | 3e-4 | 3e-4 | 6e-5 |
| Warmup / max steps | 2k / 100k | 2k / 100k | 5k / 50k |
| Best checkpoint step | 93,000 | 99,551 | 42,312 |

Checkpoints are written to `$SCRATCH` (fast, not backed up). Single-GPU and DDP saves are a `model.pt` + `optimizer.pt` + `config.yaml` triple. FSDP saves a DCP shard set (`__{rank}_0.distcp` + `meta.yaml`), which is gathered to `model.pt` before single-GPU eval.

---

## Parallelism

The interesting engineering is how the same `GPT` module is wrapped as it outgrows one GPU.

**DDP (355M).** One full replica per GPU on a single node. Gradients all-reduce after each step; optimizer state is local and unsharded. This is the straightforward data-parallel step: same effective batch as the 124M run (16 sequences), LR scaled by world size.

**FSDP (1.42B).** Parameters, gradients, and optimizer state are sharded across 16 GPUs (4 nodes × 4). A size-based auto-wrap policy (`min_num_params=100_000`) puts each attention and MLP projection in its own FSDP unit, plus `tok_emb` and `lm_head`. Layer norms stay in the root unit. Activation checkpointing keeps the 24-layer, 2048-wide model inside A100-40GB at `batch_size=2`. `torch.distributed.checkpoint` (DCP) writes one shard per rank; export reloads under the same world size and wrap policy, then gathers a full state dict for inference.

---

## FSDP silently unties weight-tied embeddings

`GPT` ties the output head to the token embedding (`lm_head.weight = tok_emb.weight`). FSDP does not preserve that alias.

PyTorch marks original parameters as already-flattened only when `use_orig_params=True`. Under the default `False`, the size-based wrap policy puts `tok_emb` and `lm_head` in separate FSDP units and flattens the shared tensor a second time as an independent `FlatParameter`. Both halves shard and synchronize correctly, so training is sound — it just trains one more `vocab × n_embd` matrix than the config claims. The 1.42B figure is that untied count.

The failure mode is at load time. A loader that unconditionally restores the tie (`model.lm_head.weight = model.tok_emb.weight`) discards the trained head and substitutes the input embedding. The signature is a trained-looking train loss against near-random val loss: here, train 0.58 vs val **7.50 / perplexity 1810**. Every conventional check passed — shapes matched, `load_state_dict` was silent, and a "unique params" counter that *assumed* tying hid the extra 103M.

The loader now decides from the checkpoint: compare the two matrices, and if they differ, allocate a fresh `lm_head` parameter *before* `load_state_dict`. On a tied model both keys address the same storage, so loading them in sequence lets the later one silently win. Single-GPU and DDP checkpoints keep the tie; the same path is a no-op for them.

Full write-up: [`docs/torch_notes.md` §13](https://github.com/rushikeshdhumal/amarel-llm-foundry/blob/feature/04-eval-closure/docs/torch_notes.md).

Any invariant a module establishes at construction — weight tying, buffer aliasing, parameter groups — can break silently under FSDP. Assert it after wrapping rather than assuming it at load time.

---

## Repository layout

Each scaling stage lives on its own branch. Branches are not merged; `main` is this overview plus shared design notes (`instructions/`, `configs/base_config.yaml`, `slurm/common.sh`). Later stages reuse earlier modules (`model.py`, `data_utils.py`, training utilities) rather than reimplementing them.

| Branch | Contents |
| :--- | :--- |
| [`feature/00-hpc-distributed-baseline`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/00-hpc-distributed-baseline) | SLURM + `torch.distributed` / NCCL smoke tests across Amarel nodes |
| [`feature/01-nanogpt-transformer`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/01-nanogpt-transformer) | `GPT` module, tokenizer, single-GPU training and generation |
| [`feature/02-ddp-multi-gpu`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/02-ddp-multi-gpu) | `DistributedDataParallel` on one node |
| [`feature/03-fsdp-hpc-sharding`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/03-fsdp-hpc-sharding) | FSDP, DCP checkpoints, multi-node training |
| [`feature/04-eval-closure`](https://github.com/rushikeshdhumal/amarel-llm-foundry/tree/feature/04-eval-closure) | Val pretokenize, DCP → `model.pt` export, eval harness, GPT-2 benchmark |

```bash
git clone https://github.com/rushikeshdhumal/amarel-llm-foundry.git
cd amarel-llm-foundry
git checkout feature/03-fsdp-hpc-sharding   # pick a stage
```

Each feature-branch `README.md` has run commands and measured results for that stage.

---

## Running on Amarel

**Prerequisites:** Amarel account, conda env with PyTorch + CUDA, dataset on scratch.

```bash
module use /projects/community/modulefiles
module load cuda/12.8.1
conda create -n llm python=3.11 -y && conda activate llm
pip install -e .

export SCRATCH=/scratch/$USER
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt \
     -P $SCRATCH/data/tinystories/
```

Jobs (details and flags are on the corresponding branch README):

```bash
cd /scratch/$USER/amarel-llm-foundry

sbatch slurm/train_1gpu.sh          # 124M, 1 GPU
sbatch slurm/train_ddp_4gpu.sh      # 355M, 4 GPUs
sbatch slurm/train_fsdp_4nodes.sh   # 1.42B, 4 nodes × 4 GPUs

sbatch slurm/pretokenize_val.sh     # build val.bin
sbatch slurm/eval.sh                # eval + GPT-2 benchmark
```

FSDP checkpoints need export to `model.pt` before single-GPU eval (`slurm/export_checkpoint.sh`, same world size as training). Checkpoints and data stay on `$SCRATCH`; copy keepers to `$HOME` — scratch is purged.

Weights are not in this repository. Best checkpoints, `report.md`, and eval JSON live on Amarel under `~/checkpoints/`.

---

## Documentation

`docs/` is not on `main`; links below point at the eval branch, which carries the most complete copy.

| Doc | Purpose |
| :--- | :--- |
| [`docs/slurm-guide.md`](https://github.com/rushikeshdhumal/amarel-llm-foundry/blob/feature/04-eval-closure/docs/slurm-guide.md) | SLURM, `torchrun`, multi-node rendezvous, Amarel partitions |
| [`docs/torch_notes.md`](https://github.com/rushikeshdhumal/amarel-llm-foundry/blob/feature/04-eval-closure/docs/torch_notes.md) | PyTorch gotchas: OOM, NCCL, FSDP weight tying, `torch.compile` |
| [`docs/maintenance_schedule.md`](https://github.com/rushikeshdhumal/amarel-llm-foundry/blob/feature/04-eval-closure/docs/maintenance_schedule.md) | Amarel maintenance windows and RHEL 9 notes |

---

## Limitations

This is a cluster training stack, not a general-purpose language model. The 1.42B checkpoint is undertrained for its size; generation does not stop at `<|endoftext|>` (fixed token count); and the GPT-2 comparison is in-domain only. Closing those gaps — a web-scale corpus, a Chinchilla-complete token budget, EOS-aware decoding — is straightforward follow-on work, not a prerequisite for showing that a 1.42B GPT trains stably across four Amarel nodes under FSDP.

---

## License

MIT — see [`LICENSE`](LICENSE).
