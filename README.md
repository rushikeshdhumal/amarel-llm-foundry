# Phase 1 — NanoGPT Transformer

**Branch**: `feature/01-nanogpt-transformer`  
**Project**: [Amarel LLM Foundry](https://github.com/rd1113/amarel-llm-foundry) — 5-phase journey from SLURM hello-world to a multi-agent system.

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
| `configs/base_config.yaml` | Global defaults (`log_interval`, `save_interval`, `seq_len`, …) |
| `configs/phase1_124M.yaml` | Phase overrides: `n_layer=12`, `n_head=12`, `n_embd=768`, `bs=16`, `lr=3e-4` |
| `slurm/train_1gpu.sh` | Single-GPU SLURM job (16h wall clock, 48 GB RAM) |
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
| `use_amp` | true | BF16/FP16 mixed precision via `torch.autocast` |

### Training log

| Run | Steps | Best loss | Final loss | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Run 1 | 0 → 27,500 | **1.2557** @ step 26,600 | 1.4173 | Killed by 4h wall-clock limit |
| Run 2 | 27,500 → 100,000 | **1.0330** @ step 91,900 | 1.1480 | Completed full cosine schedule |

Best checkpoint: `step_0091900` — use this for Phase 2 (DDP).

### Acceptance criteria

- [x] Forward pass runs without shape errors (`bs=4, seq=1024`)
- [x] Checkpoints saved with `model.pt`, `optimizer.pt`, `config.yaml`
- [x] Training script accepts `--config` and `--resume` flags
- [x] Full 100k-step run converges (best loss **1.0330**)
- [ ] Single-batch overfit test: loss < 0.01 within 100 steps
- [ ] Gradient norms logged every `grad_norm_log_interval` steps

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

### Inspect output without reading 100k lines

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
│   └── hello_*.sh             # phase 0 distributed smoke tests
└── src/
    ├── tokenizer.py
    ├── data_utils.py
    ├── model.py
    └── train_single_gpu.py
```

---

## Phase 0 — HPC Distributed Baseline ✅

> Completed on branch `feature/00-hpc-distributed-baseline`.  
> Validated `torch.distributed` + NCCL across Amarel GPU nodes.

| Test | Result |
| :--- | :--- |
| `hello_1node_1gpu.sh` | ✅ `[Rank 0/1] Host: gpu030  device=NVIDIA L40S` |
| `hello_1node_4gpu.sh` | ✅ Ranks 0–3 on `gpuk002` (A100-PCIE-40GB) |
| `hello_2nodes_4gpu.sh` | 🔄 Resubmitted after NCCL interface fix (`^lo,^docker`) |
