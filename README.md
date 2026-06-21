# Phase 1 — NanoGPT Transformer

**Branch**: `feature/01-nanogpt-transformer`  
**Project**: [Amarel LLM Foundry](https://github.com/rushikeshdhumal/amarel-llm-foundry) — 5-phase journey from SLURM hello-world to a multi-agent system.

---

## Goal

Implement a GPT-2 124M decoder-only transformer from scratch in pure PyTorch and train it to convergence on the TinyStories dataset using a single GPU on the Amarel HPC cluster.

---

## What was built

| File | Purpose |
| :--- | :--- |
| `src/tokenizer.py` | tiktoken GPT-2 BPE wrapper (`VOCAB_SIZE=50257`, EOT token handling) |
| `src/data_utils.py` | Streaming `IterableDataset` over TinyStories `.txt` — no full RAM load |
| `src/model.py` | Full GPT: `CausalSelfAttention` → `MLP` → `TransformerBlock` → `GPT` (124.4M params) |
| `src/train_single_gpu.py` | Training loop: AMP, cosine LR schedule, grad clipping, checkpointing, `--resume` |
| `src/generate.py` | Inference script: load checkpoint, generate text with temperature + top-k sampling |
| `configs/base_config.yaml` | Global defaults (`log_interval`, `save_interval`, `seq_len`, …) |
| `configs/phase1_124M.yaml` | Phase overrides: `n_layer=12`, `n_head=12`, `n_embd=768`, `bs=16`, `lr=3e-4` |
| `slurm/train_1gpu.sh` | Single-GPU SLURM job (16h wall clock, 48 GB RAM) |
| `slurm/generate.sh` | Short inference job (15 min, 1 GPU); accepts `--checkpoint <path>` |
| `docs/torch_notes.md` | PyTorch gotchas hit during development |
| `docs/slurm-guide.md` | SLURM reference for Amarel |

---

## Model architecture

```
Token embedding  (50257 × 768)
Position embedding (1024 × 768)
        ↓
 × 12  TransformerBlock
        ├── LayerNorm
        ├── CausalSelfAttention  (12 heads, head_dim=64, causal mask)
        ├── LayerNorm
        └── MLP  (768 → 3072 → 768, GELU)
        ↓
LayerNorm → Linear (768 → 50257, weight-tied to token embedding)
```

Total parameters: **124.4M**

---

## Training

### Hyperparameters

| Param | Value | Reason |
| :--- | :--- | :--- |
| `batch_size` | 16 | Attention is O(B×T²); bs=64 OOMs on 40 GB at seq=1024 |
| `seq_len` | 1024 | Full GPT-2 context window |
| `learning_rate` | 3e-4 | AdamW with cosine decay to 0 |
| `grad_clip` | 1.0 | `clip_grad_norm_` before every optimizer step |
| `use_compile` | false | Cluster GCC 4.8.5 can't build TorchInductor (`stdatomic.h`) |
| `use_amp` | true (auto) | Enabled whenever CUDA is available; not a config key — auto-detected from device |

### Training log

| Run | Steps | Best loss | Final loss | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Run 1 | 0 → 27,500 | **1.2557** @ step 26,600 | 1.4173 | Killed by 4h wall-clock limit |
| Run 2 | 27,500 → 100,000 | **1.0330** @ step 91,900 | 1.1480 | Completed full cosine schedule |

### Checkpoint note

The best loss (1.0330) occurred at step 91,900, which is **not** a multiple of `save_interval=500`.
The nearest saved checkpoints are `step_0091500` and `step_0092000`; check their `config.yaml`
for `_loss` and use the lower one.

> **Fix for future runs:** `train_single_gpu.py` now maintains a single `best/` directory inside
> the run folder that is overwritten whenever a new lowest loss is reached — regardless of
> `save_interval`. Phase 2+ runs will always have an exact best checkpoint.

### Acceptance criteria

- [x] Forward pass runs without shape errors (`bs=4, seq=1024`)
- [x] Checkpoints saved with `model.pt`, `optimizer.pt`, `config.yaml`
- [x] Training script accepts `--config` and `--resume` flags
- [x] Full 100k-step run converges (best loss **1.0330**)
- [x] Single-batch overfit test: loss drops to **0.0028** in 100 steps (proxy model)
- [x] Gradient norms logged every `grad_norm_log_interval` steps (steady ~0.42 throughout run)
- [x] Inference test: model produces coherent short story fragments — see samples below

### Inference samples

Checkpoint: `step_0093000` (loss 1.0527) · `temperature=0.8` · `top_k=40` · `max_new_tokens=150`

**Prompt 1:** *Once upon a time there was a little girl named Lily*
> Lily loved to play in her room with all the toys. One day she was playing with her doll when she
> heard a strange noise. It was loud and scary. Lily asked her mom, "What is that noise?" Her mom
> said, "It's the wind, Lily." … Lily was scared but she was brave. She went outside and saw the
> wind blowing really hard. But she was careful and she was not scared. Lily and her mom played
> until the wind stopped. `<|endoftext|>`

**Prompt 2:** *One day, a small dog found a*
> big, red ball. The dog wanted to play with the ball, but it was too big. … Then, a clever cat
> came by and saw the dog. The cat had an idea. … The dog put the stick under the ball. Now, the
> ball was easier to get! The dog and the cat played with the big, red ball all day. They were very
> happy and became good friends.

**Prompt 3:** *The sun was shining and*
> birds were singing. The boy saw a big red ball near a tree. He wanted to play with it. … "Don't
> worry, the tree is not broken. It can heal." They went to the tree and the boy's mommy put a
> small stick under the tree's roots. The tree started to dance. The tree's roots healed …

**Observations:**
- All three stories are grammatically correct and stylistically appropriate for TinyStories.
- The model learned narrative structure: setup → conflict → resolution.
- Prompt 1 ended naturally with `<|endoftext|>` — the boundary token was learned.
- No degenerate loops or random word soup at loss ~1.05.

---

## Running on Amarel

### First-time cluster setup

```bash
# On the login node
module use /projects/community/modulefiles
module load anaconda/2025.06-ts840
conda create -n llm python=3.11 -y
conda activate llm
conda install numpy -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken omegaconf datasets
```

### Download TinyStories

```bash
mkdir -p /scratch/$USER/data/tinystories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt \
     -P /scratch/$USER/data/tinystories/
```

### Submit a training job

```bash
cd /scratch/$USER/amarel-llm-foundry
git pull origin feature/01-nanogpt-transformer
sbatch slurm/train_1gpu.sh
```

### Resume from a checkpoint

```bash
sbatch slurm/train_1gpu.sh --resume $CHECKPOINT_DIR/run_<timestamp>/step_<N>
```

### Copy best checkpoint to durable storage

`$SCRATCH` is purged after 90 days and is not backed up.
`/projects/$USER` requires a separate OARC allocation request, so the Phase 1
checkpoint was saved to the home directory instead (backed up, 20 GB quota):

```bash
mkdir -p ~/checkpoints/phase1_best
cp -r $SCRATCH/checkpoints/run_<ts>/step_0093000 ~/checkpoints/phase1_best/
```

**Saved:** `~/checkpoints/phase1_best/step_0093000` (loss 1.0527)

> The true best loss (1.0330) occurred at step 91,900, which falls between two
> `save_interval=500` boundaries and was never written to disk. `step_0093000`
> is the next available checkpoint with a comparable loss (~0.02 difference).
> Phase 2+ runs will also produce a `best/` directory inside the run folder
> (overwritten whenever loss improves) so the exact best step is always captured.

### Run inference on a checkpoint

```bash
# Submit as a batch job (waits in queue):
sbatch slurm/generate.sh --checkpoint ~/checkpoints/phase1_best/step_0093000

# Or run immediately in an interactive GPU session:
srun --partition=gpu --gres=gpu:1 --mem=8G --time=00:10:00 --pty bash
cd $SLURM_SUBMIT_DIR && source slurm/common.sh
python -m src.generate \
    --checkpoint ~/checkpoints/phase1_best/step_0093000 \
    --max_new_tokens 150 --temperature 0.8 --top_k 40
```

**What to look for:** Each prompt should produce a grammatically plausible sentence or two
in the style of a children's story — not random word soup. Exact coherence depends on
how close the nearest checkpoint is to the true best loss.

### Inspect training output without reading 100k lines

```bash
OUT=gpt_1gpu_<JOBID>.out
tail -50 $OUT                                                          # did it finish?
grep "^step" $OUT | awk -F'|' '{print $2,$0}' | sort -n | head -5 | cut -d' ' -f3-  # best losses
grep -i "error\|warn\|killed\|oom" $OUT gpt_1gpu_<JOBID>.err          # problems
```

### Environment

| Item | Value |
| :--- | :--- |
| Cluster | Amarel (OARC, Rutgers University) |
| GPU | NVIDIA L40S / A100-PCIE-40GB (40 GB) |
| CUDA | 12.8.1 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 (conda env `llm`) |

---

## Repository structure

```
amarel-llm-foundry/
├── configs/
│   ├── base_config.yaml       # global defaults
│   └── phase1_124M.yaml       # phase 1 overrides
├── docs/
│   ├── slurm-guide.md         # SLURM reference for Amarel
│   └── torch_notes.md         # PyTorch gotchas
├── instructions/              # Cursor agent blueprints
├── slurm/
│   ├── common.sh              # shared env (CUDA, conda, NCCL)
│   ├── train_1gpu.sh          # phase 1 training job
│   ├── generate.sh            # inference job
│   └── hello_*.sh             # phase 0 distributed smoke tests
└── src/
    ├── tokenizer.py
    ├── data_utils.py
    ├── model.py
    ├── train_single_gpu.py
    └── generate.py
```

---

## Phase 0 — HPC Distributed Baseline ✅

> Completed on branch `feature/00-hpc-distributed-baseline`.  
> Validated `torch.distributed` + NCCL across Amarel GPU nodes.

| Test | Result |
| :--- | :--- |
| `hello_1node_1gpu.sh` | ✅ `[Rank 0/1] Host: gpu030  device=NVIDIA L40S` |
| `hello_1node_4gpu.sh` | ✅ Ranks 0–3 on `gpuk002` (A100-PCIE-40GB) |
| `hello_2nodes_4gpu.sh` | ✅ Ranks 0–3 across `gpuk002`+`gpuk003` (A100-PCIE-40GB) — NCCL 2.21.5 |
