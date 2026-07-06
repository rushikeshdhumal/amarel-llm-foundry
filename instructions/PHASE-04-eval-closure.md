# PHASE 04: Evaluation & Project Closure

**Branch**: `feature/04-eval-closure`  
**Prerequisite**: `feature/03-fsdp-hpc-sharding` (FSDP training works; Phase 1–2 checkpoints exist). Phase 3 export is optional but recommended if a 1.3B DCP checkpoint is available.  
**Goal**: Measure existing checkpoints with a reproducible eval harness, benchmark against GPT-2 baselines, archive results, and **close the project**. There is no Phase 5 and **no new large-scale training** in Phase 4.

---

## 1. Objective

Phases 0–3 built and scaled GPT models on Amarel (124M → 350M → 1.3B; DDP → FSDP). Phase 4 answers: **how good are those models, and how do they compare to GPT-2 at matched sizes?**

| # | Deliverable | Purpose |
|:-:|:------------|:--------|
| 1 | **Val set + pretokenize** | Fixed TinyStories holdout for reproducible perplexity |
| 2 | **`src/eval.py`** | Val loss, perplexity, bits per token on exported checkpoints |
| 3 | **`src/benchmark.py`** | Side-by-side val metrics + generation vs Hugging Face GPT-2 |
| 4 | **`export_fsdp_checkpoint.py`** | DCP → `model.pt` so 1.3B checkpoints work on one GPU |
| 5 | **Results README** | Consolidated tables, samples, scaling notes from Phases 1–3 |
| 6 | **Archive** | Copy best exports + eval artifacts to `~/checkpoints/` |

**Explicitly out of scope:** OpenWebText pretrain, multi-epoch long runs, multi-agent systems, fine-tuning, new training configs.

**Why TinyStories val (not OpenWebText):** All project checkpoints were trained on TinyStories. In-domain val perplexity is meaningful. GPT-2 was trained on WebText — comparisons are **architecture- and tokenizer-fair**, not training-data-fair. State that clearly in the README footnote.

---

## 2. Files to Create / Modify

| File | Action | Notes |
| :--- | :--- | :--- |
| `scripts/pretokenize_val.py` | Create | Build `val.bin` from TinyStories val split (uint16 memmap); write `val_meta.yaml`; use `src.tokenizer.Tokenizer` with `add_eot=False` (same as `TinyStoriesDataset`) |
| `scripts/export_fsdp_checkpoint.py` | Create | DCP → `model.pt` + `config.yaml`; **same world_size + FSDP wrap as training**; flatten `meta.yaml` nested config for `generate.py` |
| `src/eval.py` | Create | Single-GPU eval on `val.bin`; reuse `load_model_from_checkpoint` from `generate.py` |
| `src/benchmark.py` | Create | Orchestrate full comparison: read `eval_checkpoints.yaml` + prior JSON results; load **one** HF GPT-2 model at a time for val ppl + generation → `report.md` |
| `configs/eval_suite.yaml` | Create | `val_path`, prompts, GPT-2 model list, sampling params |
| `configs/eval_checkpoints.yaml` | Create | Paths to Phase 1 / 2 / 3 exported checkpoints for batch eval |
| `slurm/pretokenize_val.sh` | Create | CPU job: download valid split if missing, build `val.bin`; source `slurm/common.sh` |
| `slurm/export_checkpoint.sh` | Create | Multi-GPU FSDP export |
| `slurm/eval.sh` | Create | Single-GPU: `eval.py` + `benchmark.py` |
| `requirements.txt` | Modify | Add `transformers>=4.36.0`, `omegaconf`, `tiktoken`, `pyyaml` (keep in sync with `pyproject.toml`) |
| `pyproject.toml` | Modify | Add `transformers>=4.36.0` |
| `README.md` | Modify | Phase 4 results, project-complete summary, merge to `main` |
| `docs/llm_computations.md` | Modify | Short § on perplexity, bits per token, comparison caveats |

**Do not create:** `data_openwebtext.py`, OpenWebText download scripts, training SLURM scripts, or agent code.

**Optional (no new code):** Document Phase 3 **scaling efficiency** from existing `.out` logs in README (`efficiency = tok/s_4node / (tok/s_1node × 4)`).

---

## 3. Validation Dataset

### Layout

```
$SCRATCH/data/tinystories/
  TinyStoriesV2-GPT4-train.txt     # existing (Phases 1–3)
  TinyStoriesV2-GPT4-valid.txt     # download if not present
  tokenized/
    val.bin                        # uint16 token IDs
    val_meta.yaml                  # val_tokens, source, timestamp
```

### Val split

- **Preferred:** official `TinyStoriesV2-GPT4-valid.txt` from the same Hugging Face repo as train:

  ```bash
  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt \
       -P $SCRATCH/data/tinystories/
  ```

- **Fallback:** hold out the last 1% of documents from the train file (document once in `val_meta.yaml`).

### Pretokenize rules

Match training tokenization exactly (`src/data_utils.py`):

- Read the `.txt` line-by-line; skip empty lines.
- `Tokenizer.encode(line, add_eot=False)` — lines already contain `<|endoftext|>` which tiktoken maps via `allowed_special`.
- Concatenate all token IDs into one contiguous stream, then write as uint16 memmap.

### Format

```python
# val.bin: uint16 little-endian token IDs (nanoGPT convention)
tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
```

Eval uses **non-overlapping** windows: slice `(x, y)` the same way as training (`x = chunk[:seq_len]`, `y = chunk[1:seq_len+1]`), advancing by `seq_len + 1` tokens per window. Read `seq_len` from the checkpoint's `config.yaml` (default 1024). Cap total tokens evaluated at `--max_tokens` (default 1M) for speed.

---

## 4. Checkpoints to Evaluate

| Phase | Checkpoint dir | Params | HF baseline | Arch match | Export needed? |
| :---: | :--- | ---: | :--- | :--- | :---: |
| 1 | `run_<ts>/best` (or lowest-loss `step_*`) | ~124M | `gpt2` | ✅ 12L / 768D | No — has `model.pt` |
| 2 | `run_<ts>/best` | ~350M | `gpt2-medium` | ✅ 24L / 1024D | No — has `model.pt` |
| 3 | `run_<ts>/best` (DCP) | ~1.3B | `gpt2-xl` | ⚠️ param-count only (24L×2048D vs 48L×1600D) | **Yes** — export first |

**Checkpoint layouts (do not confuse them):**

| Phase | Directory contents |
| :--- | :--- |
| 1–2 | `model.pt`, `optimizer.pt`, `config.yaml` — works with `generate.py` today |
| 3 (FSDP) | DCP shard files + `meta.yaml` only — **no** `model.pt`; export before eval/generate |

Point `eval_checkpoints.yaml` at dirs that contain `model.pt` + `config.yaml` (Phase 1–2: `.../best`; Phase 3: `.../best_export` after export).

If `best/` is missing on an older Phase 1 run, use the lowest-loss `step_*` directory instead (same layout).

---

## 5. Evaluation Harness

### `src/eval.py`

```bash
python -m src.eval \
  --checkpoint $SCRATCH/checkpoints/run_<ts>/best \
  --val_path $SCRATCH/data/tinystories/tokenized/val.bin \
  --max_tokens 1000000
```

**Metrics:**

| Metric | Formula |
| :--- | :--- |
| Val loss | Mean CE over non-overlapping `seq_len` windows |
| Perplexity | `exp(val_loss)` |
| Bits per token | `val_loss / ln(2)` |

- `torch.no_grad()`, `model.eval()`.
- Use `model(x, y)` cross-entropy (same as training), not a separate loss path.
- Write `$SCRATCH/eval_results/<checkpoint_name>.json`.

### `src/benchmark.py`

- `transformers` **only** for GPT-2 baselines — never for training (`@GLOBAL.md`).
- Reads `configs/eval_checkpoints.yaml` and `$SCRATCH/eval_results/*.json` for project models.
- Load **one** HF model at a time → val eval on the same `val.bin` (same windowing + tiktoken) + generation → `del model` → next.
- Output: `$SCRATCH/benchmark_results/report.md` (perplexity table + prompt completions).

**Comparison footnote (required in report):**  
*"Our models: TinyStories pretrain. GPT-2: WebText. Same GPT-2 BPE tokenizer; Phase 3 vs gpt2-xl is param-count approximate, not architecture-matched. Not matched on training tokens or corpus."*

### `configs/eval_suite.yaml`

```yaml
val_path: ???   # $SCRATCH/data/tinystories/tokenized/val.bin

prompts:
  - "Once upon a time there was a little girl named Lily"
  - "The sun was shining and"
  - "In a surprising turn of events, the company announced"

sampling:
  max_new_tokens: 150
  temperature: 0.8
  top_k: 40

gpt2_models:    # loaded one at a time
  - gpt2
  - gpt2-medium
  - gpt2-large
  - gpt2-xl
```

Use TinyStories-style prompts for your models; optional neutral web-style prompts for GPT-2-only section with a domain caveat.

### `configs/eval_checkpoints.yaml`

List exported checkpoint dirs (must each contain `model.pt` + `config.yaml`):

```yaml
checkpoints:
  - name: phase1_124M
    path: ???   # $SCRATCH/checkpoints/run_<phase1_ts>/best
  - name: phase2_350M
    path: ???   # $SCRATCH/checkpoints/run_<phase2_ts>/best
  # Optional — only after FSDP export:
  - name: phase3_1B
    path: ???   # $SCRATCH/checkpoints/run_<phase3_ts>/best_export
```

---

## 6. FSDP Checkpoint Export

Required before eval/generate on Phase 3 DCP checkpoints.

### `scripts/export_fsdp_checkpoint.py`

- Rebuild model with **same FSDP policy** as `train_fsdp.py`: `size_based_auto_wrap_policy(min_num_params=100_000)`, `ShardingStrategy.FULL_SHARD`, same `MixedPrecision`.
- All ranks `dcp.load()` the checkpoint (model + optimizer shards — collective required even though export only writes weights).
- Gather with `FSDP.state_dict_type(..., FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))`.
- Rank 0 writes `<ckpt_dir>_export/model.pt` and a flat `config.yaml` (copy fields from `meta.yaml`'s nested `config`, plus `_step` / `_loss` from metadata — same shape as Phase 1–2 checkpoints).
- Launch with **same `world_size` and node layout** as the training job that wrote the checkpoint. Default Phase 3 job: 4 nodes × 4 GPUs = 16 (`slurm/train_fsdp_4nodes.sh`). If you ran a 1-node baseline (`sbatch --nodes=1 ...`), export with `--nodes=1` instead.
- **Mismatched world size → export fails or produces corrupt weights.**

SLURM scripts must wrap `torchrun` in `srun --label` (same pattern as `train_fsdp_4nodes.sh`) and parse `--checkpoint` via bash (same pattern as `generate.sh`):

```bash
sbatch slurm/export_checkpoint.sh --checkpoint $SCRATCH/checkpoints/run_<ts>/best
```

Export job needs multi-node CPU RAM on rank 0 (~6 GB fp32 gather with `offload_to_cpu`); use `#SBATCH --mem=80G` like training, not single-GPU eval mem.

After export, `python -m src.generate` and `src/eval` work on one GPU (`#SBATCH --mem=32G` for 1.3B inference).

---

## 7. Acceptance Criteria

### Val data
- [ ] `val.bin` + `val_meta.yaml` under `$SCRATCH/data/tinystories/tokenized/`
- [ ] Documented download / split method

### Export (Phase 3)
- [ ] If 1.3B DCP exists: exported `model.pt` + flat `config.yaml` load in `generate.py` without error
- [ ] Export job used same `--nodes` / GPU count as the **original** training job (check SLURM `.out` for `world_size=`)

### Eval
- [ ] `eval.py` runs on Phase 1, Phase 2, and Phase 3 (if exported) checkpoints
- [ ] JSON results under `$SCRATCH/eval_results/`

### Benchmark
- [ ] `benchmark.py` produces `report.md` with val perplexity for your models + GPT-2 baselines
- [ ] Generation samples for all prompts in `eval_suite.yaml`
- [ ] Training-data caveat footnote present

### Project closure
- [ ] README updated: all phases ✅, results table, scaling notes (from Phase 3 logs if available)
- [ ] Best exports + `report.md` copied to `~/checkpoints/phase{1,2,3}_best/` (or equivalent)
- [ ] `main` branch landing page reflects **project complete**

---

## 8. Amarel Notes

### SLURM argument passing

`sbatch` does not forward arbitrary flags to your script. Every Phase 4 SLURM script must parse bash arguments in a `while/case` loop (see `slurm/generate.sh` and `slurm/train_fsdp_4nodes.sh`).

### Pretokenize val first (CPU job)

```bash
sbatch slurm/pretokenize_val.sh
```

Fail fast in `eval.py` if `val.bin` is missing.

### HF cache

```bash
export HF_HOME=$SCRATCH/hf_cache
```

Pre-download `gpt2-xl` on the login node before benchmark jobs.

### Eval / benchmark SLURM

```bash
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --mem=32G        # 1.3B eval; 8G is enough for Phase 1–2 only
#SBATCH --time=01:00:00
```

Phase 1–2-only eval can reuse `slurm/generate.sh` mem settings (`--mem=8G`).

### Export SLURM

Match training topology (default: 4 nodes × 4 GPUs). Source `slurm/common.sh`; wrap `torchrun` in `srun`.

---

## 9. Running on Amarel

### Step 0 — Build val set

```bash
cd /scratch/$USER/amarel-llm-foundry
git checkout feature/04-eval-closure
sbatch slurm/pretokenize_val.sh
```

### Step 1 — Export Phase 3 (if needed)

```bash
sbatch slurm/export_checkpoint.sh \
  --checkpoint $SCRATCH/checkpoints/run_<phase3_ts>/best
```

### Step 2 — Eval + benchmark

```bash
sbatch slurm/eval.sh
# Loops eval.py over eval_checkpoints.yaml, then runs benchmark.py once
```

Or interactively:

```bash
python -m src.eval --checkpoint $SCRATCH/checkpoints/run_<ts>/best \
  --val_path $SCRATCH/data/tinystories/tokenized/val.bin

# After all eval JSON files exist:
python -m src.benchmark --config configs/eval_suite.yaml
```

Find checkpoint paths in Phase SLURM `.out` logs (`[ckpt] Saved checkpoint to ...` or `Checkpoints: run_<ts>`). Phase 3 resume runs may write to a **new** `run_<ts>` — evaluate the run that holds your best DCP checkpoint.

### Step 3 — Archive + merge

```bash
cp -r $SCRATCH/checkpoints/run_<phase2_ts>/best ~/checkpoints/phase2_best/
cp $SCRATCH/benchmark_results/report.md ~/checkpoints/
# Merge feature/04-eval-closure → main
```

---

## 10. Expected Results (indicative)

Document **actual** numbers in README — these are not pass/fail gates.

| Model | Val perplexity (TinyStories val, indicative) |
| :--- | :--- |
| Phase 1 ~124M | ~15–25 |
| Phase 2 ~350M | ~8–15 |
| Phase 3 ~1.3B | ~6–12 |
| `gpt2` / `gpt2-medium` | Architecture-matched baselines; on TinyStories val they may look **worse** than your in-domain models — explain domain mismatch |
| `gpt2-xl` vs Phase 3 | Param-count comparison only; do not claim architecture parity |

Success = reproducible metrics, honest comparison footnote, coherent generation samples (Phase 2+ already demonstrated), polished README.

**Rough timeline:** pretokenize ~30 min (CPU); export ~15–30 min (16 GPUs, if Phase 3 exists); eval + benchmark ~1–2 h (single GPU). No multi-day training.

---

## 11. Key Differences from Prior Phase 4 Plan

| Prior plan | This plan |
| :--- | :--- |
| OpenWebText pretrain, 300k steps | **No new training** |
| Weeks of 24h resume chains | **Days** (CPU pretokenize + single-GPU eval + optional export) |
| Milestone DCP copies | **Eval existing checkpoints only** |
| Multi-agent capstone | **Eval + documentation closure** |

---

## 12. Cursor Instructions

- Pure `torch.nn` for your models; `transformers` only in `benchmark.py`.
- Import or factor `load_model_from_checkpoint` from `generate.py` into `eval.py` (do not duplicate loading logic).
- Type hints; module-level imports (`docs/torch_notes.md` §1).
- Do not add training scripts or OpenWebText data pipelines.
- Do not modify `src/data_utils.py` or training configs — Phase 4 is eval-only.
- README is the primary Phase 4 artifact — make results scannable (table + 2–3 generation blocks).

---

## 13. Project Complete

When acceptance criteria pass:

1. All four phases marked ✅ in README project map  
2. `feature/04-eval-closure` merged to `main`  
3. No further phases planned  

**Amarel LLM Foundry — complete.**
