# Phase 4 — Evaluation & Project Closure

**Branch**: `feature/04-eval-closure`  
**Project**: [Amarel LLM Foundry](https://github.com/rushikeshdhumal/amarel-llm-foundry) — a 5-phase journey from SLURM hello-world to scaled GPT training and reproducible evaluation on Amarel HPC.

---

## Goal

Phases 0–3 built and scaled GPT models on TinyStories (124M → 350M → 1.4B; single-GPU → DDP → FSDP). Phase 4 answers: **how good are those checkpoints, and how do they compare to open-source GPT-2 at matched sizes?**

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
| 3 | `run_<ts>/best` (DCP) | ~1.4B | `gpt2-xl` | ⚠️ param-count only | **Yes** |

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

All metrics from job `60657096` (1M tokens, 976 non-overlapping windows).

| Model | Train loss (ckpt) | Val loss | Perplexity | Bits/token |
| :--- | :---: | :---: | :---: | :---: |
| Phase 1 ~124M (step 93000) | 1.0527 | 1.0848 | 2.96 | 1.5650 |
| Phase 2 ~350M (step 99551) | 0.7180 | 0.9460 | 2.58 | 1.3648 |
| **Phase 3 ~1.4B (step 42312)** | 0.5811 | **0.9311** | **2.54** | **1.3433** |
| `gpt2` | WebText | 2.4708 | 11.83 | 3.564 |
| `gpt2-medium` | WebText | 2.1944 | 8.97 | 3.166 |
| `gpt2-large` | WebText | 2.0467 | 7.74 | 2.953 |
| `gpt2-xl` | WebText | 1.9625 | 7.12 | 2.831 |

TinyStories val perplexity in the ~2.5–3 range is expected for in-domain models. Every phase beats every GPT-2 size on this data; GPT-2 improves with scale but stays behind, which is domain mismatch (WebText vs TinyStories), not a failure of GPT-2.

Scaling holds monotonically (1.0848 → 0.9460 → 0.9311) but flattens sharply between Phase 2 and Phase 3 — 4× the parameters buys 0.015 nats. Two reasons, both about the training budget rather than the architecture:

- **Phase 3 saw far less data.** It stopped at step 42312 of 50000 with `batch_size=2` per GPU across 16 GPUs — roughly 1.4B tokens, against Phase 2's 99551 steps. The 1.4B model is undertrained for its size, so most of its extra capacity is unused.
- **The train-loss column overstates Phase 3.** `save_best_only` selects on a single mini-batch loss from rank 0, and at `batch_size=2` that estimate is very noisy, so "best" tends to pick a lucky batch. This biases 0.5811 low; the 0.93 val loss is the honest number. Phase 1 and 2 used larger per-rank batches and are less affected.

Neither undermines the phase objective, which was to demonstrate that a 1.4B model trains stably across 4 nodes under FSDP. Closing the gap to Phase 2 would need more steps and a larger effective batch, not a different model.

### Generation samples

Excerpts from `$SCRATCH/benchmark_results/report.md` (temperature 0.8, top-k 40). Completions are truncated here; the full text for every prompt and model is in the report.

**In-domain prompt — all three phases produce coherent TinyStories prose:**

> **Prompt:** Once upon a time there was a little girl named Lily

| Model | Completion |
| :--- | :--- |
| Phase 1 ~124M | …Lily loved playing outside in the sunshine, and she was always very alert. Every day she would take lots of walks and explore the world. One day, Lily was walking through a field and saw a big tree with lots of leaves. She thought it looked interesting, so she decided to climb it… |
| Phase 2 ~350M | …She was very excited because it was her birthday. Lily had a big present waiting for her. It was wrapped in shiny paper with a big bow on top. Lily couldn't wait to open the present, so she opened it up very carefully… |
| Phase 3 ~1.4B | …She was a very lucky girl and she had a very special toy — a silver toy car. She loved to drive her car around and around. One day, Lily's mommy gave her a big, shiny silver ball… |

Grammar and narrative structure are solid at every scale. The differences are subtle: all three stay on-genre, and none of them is obviously better as a story. This matches the perplexity table, where Phase 2 → Phase 3 moved only 0.015 nats.

**Out-of-domain prompt — where the TinyStories models fail and GPT-2 does not:**

> **Prompt:** In a surprising turn of events, the company announced

| Model | Completion |
| :--- | :--- |
| Phase 3 ~1.4B | …, "We have a special gift for you, Sue!" Sue ran to the other side of the park and saw a huge pile of presents wrapped in colorful paper… |
| `gpt2-xl` | …on Monday that it would stop selling the Lumia 920 and the Lumia 820 on Friday, November 19. The company did not go into details of the reason for the discontinuation… |

This is the qualitative counterweight to the perplexity table. Our models beat every GPT-2 size on TinyStories val loss, but only because the eval is in-domain: given a business-news prompt, Phase 3 immediately reverts to a children's story, while `gpt2-xl` continues in register. **Perplexity on a narrow corpus measures fit to that corpus, not general capability.**

**Two artefacts worth noting:**

- Our completions run past `<|endoftext|>` and start a new, unrelated story. `generate.py` samples a fixed `max_new_tokens` and does not stop at the EOS token — a generation-loop limitation, not a model defect.
- Sentence boundaries are missing spaces (`explore the world.One day`) because the training pipeline strips newlines, which in TinyStories carry the paragraph break. All three phases reproduce it, which is the models learning the corpus faithfully — artefacts included.

---

## Acceptance criteria

### Val data
- [x] `val.bin` + `val_meta.yaml` under `$SCRATCH/data/tinystories/tokenized/` (5,369,522 tokens)
- [x] Download / split method documented (official `TinyStoriesV2-GPT4-valid.txt`)

### Export (Phase 3, if checkpoint exists)
- [x] Export job used same `--nodes` / GPU count as the original training job (16 GPUs)
- [x] Export verified: 221 keys, 1.416B params, `lm_head` untied from `tok_emb` as FSDP trained it

### Eval
- [x] `eval.py` runs on Phase 1, Phase 2, and Phase 3 checkpoints
- [x] JSON results under `$SCRATCH/eval_results/` (job `60657096`)

### Benchmark
- [x] `benchmark.py` produces `report.md` with val perplexity + generation samples (job `60657096`)
- [x] Training-data caveat footnote present

### Project closure
- [x] Results table filled in above
- [x] Generation samples pasted in above from `report.md`
- [ ] Best exports + `report.md` copied to `~/checkpoints/phase{1,2,3}_best/`
- [ ] Phase 4 summary reflected on `main` (each phase keeps its own branch — nothing is merged)

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

Phase 3 completed at step 42312 (loss 0.5811), 16 DCP shards under `run_20260722_182718/best/`:

```bash
sbatch slurm/export_checkpoint.sh \
  --checkpoint $SCRATCH/checkpoints/run_20260722_182718/best
```

Output: `$SCRATCH/checkpoints/run_20260722_182718/best_export/` with `model.pt` + `config.yaml`. The checkpoint has 16 shards (`__0_0.distcp` … `__15_0.distcp`), matching the script's default 4 nodes × 4 GPUs — no `--nodes` override needed.

### Step 2 — Eval + benchmark

`configs/eval_checkpoints.yaml` already points at all three phases:

| Phase | Path |
| :--- | :--- |
| 1 (~124M) | `run_20260619_182306/step_0093000` (no `best/` on this run) |
| 2 (~350M) | `run_20260628_022929/best` |
| 3 (~1.4B) | `run_20260722_182718/best_export` (after Step 1 export) |

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

`$SCRATCH` is purged periodically, so the final artefacts move to `$HOME`. Note the trailing-slash trap: `cp -r SRC DEST/` nests a subdirectory when `DEST` already exists, so name the destination explicitly and create the parent first.

```bash
mkdir -p ~/checkpoints

cp -rT $SCRATCH/checkpoints/run_20260619_182306/step_0093000 ~/checkpoints/phase1_best
cp -rT $SCRATCH/checkpoints/run_20260628_022929/best        ~/checkpoints/phase2_best
cp -rT $SCRATCH/checkpoints/run_20260722_182718/best_export ~/checkpoints/phase3_best

cp $SCRATCH/benchmark_results/report.md ~/checkpoints/
cp -rT $SCRATCH/eval_results ~/checkpoints/eval_results
```

Phase 3's `model.pt` is ~5.7 GB (1.416B fp32 params plus mask buffers), so check `quota` before copying. Verify afterwards:

```bash
du -sh ~/checkpoints/*
ls ~/checkpoints/phase3_best      # expect model.pt + config.yaml
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

Each phase lives permanently on its own branch; branches are never merged. `main` carries only the project summary, so the full history of a phase stays readable in isolation.

| Phase | Branch | Status | Goal |
| :--- | :--- | :---: | :--- |
| 0 | `feature/00-hpc-distributed-baseline` | ✅ | Validate `torch.distributed` + NCCL across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | ✅ | 124M GPT from scratch, single-GPU |
| 2 | `feature/02-ddp-multi-gpu` | ✅ | DDP 4-GPU, 350M model — coherent text confirmed |
| 3 | `feature/03-fsdp-hpc-sharding` | ✅ | FSDP 4-node, 1.4B model |
| **4** | `feature/04-eval-closure` | 🔄 | Eval harness, GPT-2 benchmark, project closure |

**Amarel LLM Foundry — complete when Phase 4 is finished on its branch and `main`'s summary is updated.**
