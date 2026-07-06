# Phase 4 — Evaluation & Project Closure

**Branch**: `feature/04-eval-closure`  
**Project**: [Amarel LLM Foundry](https://github.com/rushikeshdhumal/amarel-llm-foundry) — a 5-phase journey from SLURM hello-world to scaled GPT training and reproducible evaluation on Amarel HPC.

---

## Goal

Phases 0–3 built and scaled GPT models on TinyStories (124M → 350M → 1.3B; single-GPU → DDP → FSDP). Phase 4 answers: **how good are those checkpoints, and how do they compare to open-source GPT-2 at matched sizes?**

There is **no new large-scale training** in this phase. Work is limited to:

1. A fixed TinyStories validation set (`val.bin`) for reproducible perplexity
2. Single-GPU eval on exported checkpoints (Phase 1–2 native; Phase 3 after DCP export)
3. Benchmark against Hugging Face GPT-2 baselines (`transformers` for baselines only)
4. Archive results and close the project

---

## What was built

| File | Purpose |
| :--- | :--- |
| `scripts/pretokenize_val.py` | Tokenize TinyStories valid split → `val.bin` + `val_meta.yaml` |
| `scripts/export_fsdp_checkpoint.py` | Gather FSDP DCP shards → `model.pt` + `config.yaml` for single-GPU use |
| `src/eval.py` | Val loss, perplexity, bits per token on `val.bin` |
| `src/benchmark.py` | Full comparison report: project checkpoints + GPT-2 baselines |
| `configs/eval_suite.yaml` | Val path, prompts, sampling params, GPT-2 model list |
| `configs/eval_checkpoints.yaml` | Paths to Phase 1 / 2 / 3 exported checkpoints |
| `slurm/pretokenize_val.sh` | CPU job: download valid split if missing, build `val.bin` |
| `slurm/export_checkpoint.sh` | Multi-GPU FSDP export (same topology as training) |
| `slurm/eval.sh` | Single-GPU: loop `eval.py`, then `benchmark.py` |

Shared from earlier phases: `src/generate.py` (checkpoint loading), `src/model.py`, `src/tokenizer.py`.

**Out of scope:** OpenWebText pretrain, multi-agent systems, fine-tuning, new training configs.

---

## Checkpoints to evaluate

| Phase | Checkpoint | Params | HF baseline | Arch match | Export? |
| :---: | :--- | ---: | :--- | :--- | :---: |
| 1 | `run_<ts>/best` (or lowest-loss `step_*`) | ~124M | `gpt2` | ✅ 12L / 768D | No |
| 2 | `run_<ts>/best` | ~350M | `gpt2-medium` | ✅ 24L / 1024D | No |
| 3 | `run_<ts>/best` (DCP) | ~1.3B | `gpt2-xl` | ⚠️ param-count only | **Yes** |

**Checkpoint layouts:**

| Phase | Directory contents |
| :--- | :--- |
| 1–2 | `model.pt`, `optimizer.pt`, `config.yaml` — works with `generate.py` today |
| 3 (FSDP) | DCP shard files + `meta.yaml` only — export to `best_export/` before eval |

Point `configs/eval_checkpoints.yaml` at dirs containing `model.pt` + `config.yaml`.

---

## How evaluation works

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 0 — Pretokenize (CPU)                                             │
│    TinyStoriesV2-GPT4-valid.txt  →  val.bin (uint16 memmap)             │
│    Tokenization matches training: Tokenizer.encode(line, add_eot=False) │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1 — Export Phase 3 (optional, multi-GPU)                          │
│    DCP shards  →  FullStateDict gather  →  best_export/model.pt         │
│    Requires same world_size as the training job (default: 16 GPUs)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2 — Eval (single GPU)                                             │
│    For each checkpoint in eval_checkpoints.yaml:                        │
│      load model.pt  →  non-overlapping (x, y) windows on val.bin        │
│      metrics: val_loss, perplexity = exp(loss), bits/token = loss/ln(2) │
│      write $SCRATCH/eval_results/<name>.json                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3 — Benchmark (single GPU, one HF model at a time)              │
│    Project JSON results + GPT-2 baselines on same val.bin               │
│    Generation samples for prompts in eval_suite.yaml                    │
│    write $SCRATCH/benchmark_results/report.md                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Comparison caveat (required in report):**  
*Our models: TinyStories pretrain. GPT-2: WebText. Same GPT-2 BPE tokenizer; Phase 3 vs gpt2-xl is param-count approximate, not architecture-matched. Not matched on training tokens or corpus.*

---

## Results

> Fill in after running eval + benchmark. Indicative ranges on TinyStories val — not pass/fail gates.

| Model | Val loss | Perplexity | Bits/token |
| :--- | :---: | :---: | :---: |
| Phase 1 ~124M | TBD | ~15–25 | TBD |
| Phase 2 ~350M | TBD | ~8–15 | TBD |
| Phase 3 ~1.3B | TBD | ~6–12 | TBD |
| `gpt2` | TBD | — | TBD |
| `gpt2-medium` | TBD | — | TBD |
| `gpt2-xl` | TBD | — | TBD |

### Generation samples

<!-- Paste 2–3 prompt/completion blocks from report.md after benchmark runs -->

---

## Acceptance criteria

### Val data
- [ ] `val.bin` + `val_meta.yaml` under `$SCRATCH/data/tinystories/tokenized/`
- [ ] Download / split method documented

### Export (Phase 3, if checkpoint exists)
- [ ] Exported `model.pt` + flat `config.yaml` load in `generate.py` without error
- [ ] Export job used same `--nodes` / GPU count as the original training job

### Eval
- [ ] `eval.py` runs on Phase 1, Phase 2, and Phase 3 (if exported) checkpoints
- [ ] JSON results under `$SCRATCH/eval_results/`

### Benchmark
- [ ] `benchmark.py` produces `report.md` with val perplexity + generation samples
- [ ] Training-data caveat footnote present

### Project closure
- [ ] Results table and generation samples filled in above
- [ ] Best exports + `report.md` copied to `~/checkpoints/phase{1,2,3}_best/`
- [ ] Merge `feature/04-eval-closure` → `main`

---

## Running on Amarel

### Prerequisites

```bash
conda activate llm
pip install transformers>=4.36.0   # GPT-2 baselines only; add to env once
export HF_HOME=$SCRATCH/hf_cache     # cache HF model weights on scratch
```

Phase 1–2 checkpoints must exist under `$SCRATCH/checkpoints/`. After Phase 3 training completes, export the DCP `best/` checkpoint before eval.

### Step 0 — Build validation set

```bash
cd /scratch/$USER/amarel-llm-foundry
git checkout feature/04-eval-closure
sbatch slurm/pretokenize_val.sh
```

Downloads `TinyStoriesV2-GPT4-valid.txt` if missing, then writes:

```
$SCRATCH/data/tinystories/tokenized/val.bin
$SCRATCH/data/tinystories/tokenized/val_meta.yaml
```

### Step 1 — Export Phase 3

After Phase 3 FSDP training completes:

```bash
sbatch slurm/export_checkpoint.sh \
  --checkpoint $SCRATCH/checkpoints/run_<phase3_ts>/best
```

Output: `$SCRATCH/checkpoints/run_<phase3_ts>/best_export/` with `model.pt` + `config.yaml`.

Find checkpoint paths in Phase 3 SLURM `.out` logs. If training was resumed, use the `run_<ts>` that holds your best DCP checkpoint.

### Step 2 — Eval + benchmark

Edit `configs/eval_checkpoints.yaml` with your checkpoint paths, then:

```bash
sbatch slurm/eval.sh
```

Or interactively:

```bash
python -m src.eval \
  --checkpoint $SCRATCH/checkpoints/run_<ts>/best \
  --val_path $SCRATCH/data/tinystories/tokenized/val.bin

python -m src.benchmark --config configs/eval_suite.yaml
```

### Step 3 — Archive

```bash
cp -r $SCRATCH/checkpoints/run_<phase1_ts>/best ~/checkpoints/phase1_best/
cp -r $SCRATCH/checkpoints/run_<phase2_ts>/best ~/checkpoints/phase2_best/
cp -r $SCRATCH/checkpoints/run_<phase3_ts>/best_export ~/checkpoints/phase3_best/  # if exported
cp $SCRATCH/benchmark_results/report.md ~/checkpoints/
```

### Monitor output

```bash
OUT=eval_<JOBID>.out
grep -i "perplexity\|val_loss\|error" $OUT
cat $SCRATCH/benchmark_results/report.md
```

**Rough timeline:** pretokenize ~30 min (CPU); export ~15–30 min (16 GPUs, if Phase 3 exists); eval + benchmark ~1–2 h (single GPU).

---

## Key differences from Phase 3

| Aspect | Phase 3 | Phase 4 |
| :--- | :--- | :--- |
| Primary work | FSDP training (1.3B) | Eval + benchmark only |
| New GPUs needed | 16 (training) | 1 (eval); 16 only for export |
| Checkpoint format | DCP shards | `model.pt` export for inference |
| `transformers` | Not used | Baselines in `benchmark.py` only |
| Dataset | Train split stream | Fixed pretokenized val split |
| Duration | Hours–days | Hours |

---

## Repository structure

```
amarel-llm-foundry/
├── configs/
│   ├── base_config.yaml
│   ├── phase1_124M.yaml
│   ├── phase2_350M.yaml
│   ├── phase3_1B.yaml
│   ├── eval_suite.yaml          # Phase 4
│   └── eval_checkpoints.yaml    # Phase 4
├── scripts/
│   ├── pretokenize_val.py       # Phase 4
│   └── export_fsdp_checkpoint.py # Phase 4
├── slurm/
│   ├── common.sh
│   ├── pretokenize_val.sh       # Phase 4
│   ├── export_checkpoint.sh     # Phase 4
│   ├── eval.sh                  # Phase 4
│   ├── train_1gpu.sh            # Phase 1
│   ├── train_ddp_4gpu.sh        # Phase 2
│   ├── train_fsdp_4nodes.sh     # Phase 3
│   └── generate.sh
└── src/
    ├── model.py
    ├── tokenizer.py
    ├── data_utils.py
    ├── train_single_gpu.py
    ├── train_ddp.py
    ├── train_fsdp.py
    ├── generate.py
    ├── eval.py                  # Phase 4
    └── benchmark.py             # Phase 4
```

Full phase instructions: `instructions/PHASE-04-eval-closure.md`.

---

## Project map

| Phase | Branch | Status | Goal |
| :--- | :--- | :---: | :--- |
| 0 | `feature/00-hpc-distributed-baseline` | ✅ | Validate `torch.distributed` + NCCL across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | ✅ | 124M GPT from scratch, single-GPU |
| 2 | `feature/02-ddp-multi-gpu` | ✅ | DDP 4-GPU, 350M model — coherent text confirmed |
| 3 | `feature/03-fsdp-hpc-sharding` | ✅ | FSDP 4-node, 1.3B model |
| **4** | `feature/04-eval-closure` | 🔄 | Eval harness, GPT-2 benchmark, project closure |

**Amarel LLM Foundry — complete when Phase 4 merges to `main`.**
