# SLURM Guide — Amarel HPC

A concise reference for submitting and monitoring jobs on the Amarel cluster.

---

## Mental Model

```
Login node (amarel1/2)  ──sbatch──▶  SLURM queue  ──allocates──▶  compute node(s)
                                                                         │
                                                          runs your bash script here
                                                          stdout/stderr → .out/.err
```

- **Never run computation on the login node.** It is shared by all users and has no GPUs.
- `sbatch` returns immediately with a job ID. The job runs asynchronously.
- All output is written to files; there is no live terminal attached.

---

## Submitting Jobs

```bash
# Always run from the repo root
cd /scratch/$USER/amarel-llm-foundry

sbatch slurm/<script>.sh
# → Submitted batch job 12345678
```

The number returned is your **job ID**. Every file, log, and status query uses it.

---

## Key `#SBATCH` Directives

| Directive | Example | Meaning |
|---|---|---|
| `--job-name` | `hello_1n1g` | Short label shown in queue |
| `--partition` | `gpu` | Which node pool to use |
| `--nodes` | `2` | Number of nodes |
| `--ntasks-per-node` | `4` | MPI tasks / torchrun workers per node |
| `--cpus-per-task` | `4` | CPU cores per task |
| `--gres` | `gpu:2` | GPUs per node |
| `--mem` | `32G` | RAM per node |
| `--time` | `01:00:00` | Wall-clock limit (HH:MM:SS) — job is killed at this limit |
| `--output` | `%x_%j.out` | stdout file (`%x`=job name, `%j`=job ID) |
| `--error` | `%x_%j.err` | stderr file |
| `--export` | `ALL` | Forward current env vars into the job |

---

## Monitoring Jobs

```bash
# Show your jobs in the queue
squeue -u $USER

# Verbose view with wait reason
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Watch live (refreshes every 5s)
watch -n 5 squeue -u $USER
```

### Job State Codes (`ST` column)

| Code | Meaning |
|---|---|
| `PD` | Pending — waiting for free resources |
| `R` | Running |
| `CG` | Completing — cleaning up |
| `F` | Failed |
| `TO` | Timed out (hit `--time` limit) |
| *(gone)* | Finished — check output files |

### Why is my job pending? (`%R` column)

| Reason | Cause |
|---|---|
| `Resources` | Not enough free GPUs/CPUs right now — normal, just wait |
| `Priority` | Higher-priority jobs ahead in queue |
| `QOSMaxGRES` | You've hit your per-user GPU quota |
| `ReqNodeNotAvail` | Requested node is down; remove `--nodelist` if set |

---

## Reading Output

```bash
# Stream live output while job is running
tail -f hello_1n1g_<JOBID>.out

# Check errors
cat hello_1n1g_<JOBID>.err

# After job finishes — check exit code and timing
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

### Inspecting a completed training job (without reading 100k lines)

Training jobs write one log line per step. Use these targeted commands instead of reading the whole file.

```bash
OUT=<JOBNAME>_<JOBID>.out

# 1. Did it finish cleanly? See the last 50 lines.
tail -50 $OUT

# 2. Best (lowest) loss across the entire run.
#    -F'|' splits on pipe; field 2 is " loss X.XXXX "; sort -n reads the leading number.
grep "^step" $OUT | awk -F'|' '{print $2, $0}' | sort -n | head -5 | cut -d' ' -f3-

# 3. One-liner summary: first steps, last steps, best loss.
echo "=== First 3 steps ===" && grep "^step" $OUT | head -3
echo "=== Last 3 steps ===" && grep "^step" $OUT | tail -3
echo "=== Best loss ===" && grep "^step" $OUT | sort -t'|' -k2 -n | head -1

# 4. Sparse loss curve — every 10th logged step (good for a quick trend).
grep "^step" $OUT | awk 'NR % 10 == 0 {print NR, $2, $4}'

# 5. All checkpoint saves.
grep -i "checkpoint\|saved\|saving" $OUT

# 6. Any errors, warnings, or OOM events (check both .out and .err).
grep -i "error\|warn\|killed\|oom\|traceback" $OUT gpt_1gpu_<JOBID>.err
```

> **Tip:** The `.err` file is normally empty on a clean run. Any content there is worth reading immediately.

## Finding Completed Jobs (when you don't have the job ID)

`squeue` only shows active (running/pending) jobs. For completed, failed, or timed-out jobs use `sacct`, which queries the SLURM accounting database.

```bash
# All your jobs from the last 24 hours (default window)
sacct -u $USER

# Last 7 days with the most useful columns
sacct -u $USER --starttime=now-7days \
      --format=JobID,JobName,State,ExitCode,Elapsed,Start,End

# Filter by outcome
sacct -u $USER --starttime=now-7days --state=COMPLETED,FAILED,TIMEOUT
```

### `sacct` State values

| State | Meaning |
|---|---|
| `COMPLETED` | Exited cleanly (exit code 0) |
| `FAILED` | Non-zero exit code — check `.err` file |
| `TIMEOUT` | Hit the `--time` wall-clock limit |
| `CANCELLED` | Cancelled by user (`scancel`) or admin |
| `OUT_OF_MEMORY` | Job was killed by the OOM killer |

Once you find the job ID, get full detail:
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

---

## Cancelling Jobs

```bash
scancel <JOBID>          # cancel one job
scancel -u $USER         # cancel all your jobs
scancel -u $USER -n foo  # cancel all jobs named "foo"
```

---

## Interactive GPU Session

Useful for debugging before submitting a batch job:

```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
```

Once allocated you land directly on a compute node with a live shell.

---

## Project-Specific Conventions

All scripts in `slurm/` follow these rules (enforced by `slurm/common.sh`):

- **Source `common.sh` first** — loads CUDA 12.8.1, activates the `llm` conda env, sets `$SCRATCH`, exports NCCL variables.
- **Use `torchrun`**, not `mpirun`, for all PyTorch distributed launches.
- **Logs go to `$SLURM_SUBMIT_DIR/`** via `--output=%x_%j.out` (i.e. next to the script, in the repo root).
- **Checkpoints and data go to `$SCRATCH`** (`/scratch/$USER`), never to `/home/`.

### NCCL variables set in `common.sh`

| Variable | Value | Why |
|---|---|---|
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `1` | Surfaces clear errors instead of hanging (PyTorch 2.x name; old name was `NCCL_ASYNC_ERROR_HANDLING`) |
| `NCCL_IB_DISABLE` | `1` | Amarel uses Ethernet, not InfiniBand |
| `NCCL_SOCKET_IFNAME` | `^lo,^docker` | Exclude loopback/docker; let NCCL auto-select the real NIC |
| `NCCL_DEBUG` | `WARN` | Print NCCL warnings to `.err`; helps diagnose interface/port issues |
| `NCCL_TIMEOUT` | `1800` | 30-min timeout for slow cross-node init |

### Why `NCCL_SOCKET_IFNAME=eth0` is dangerous

Hardcoding an interface name like `eth0` is a common mistake. Amarel compute nodes use unpredictable NIC names (`ens`, `enp3s0`, `bond0`, etc.) that differ between node pools. If NCCL can't find `eth0`, it falls back to loopback — all ranks appear to communicate but cross-node collectives produce `ncclInvalidUsage` or silently hang.

**Observed failure (job 56728790):** 2-node job on `gpuk002` + `gpuk009` failed with `ncclInvalidUsage` on the first `dist.barrier()` because `eth0` did not exist on those nodes.

Use the exclude-prefix syntax instead — it works regardless of the actual interface name:
```bash
export NCCL_SOCKET_IFNAME=^lo,^docker   # exclude loopback and docker; use whatever remains
```

If a multi-node job still hangs, find the real interface name with an interactive session:
```bash
srun --partition=gpu --nodes=1 --gres=gpu:1 --time=00:10:00 --pty bash -c "ip -o link show | awk -F': ' '{print \$2}'"
```
Then pin it explicitly, e.g. `NCCL_SOCKET_IFNAME=enp3s0`.

---

## Job Dependencies

SLURM makes **no guarantee** about submission order. Each job is scheduled independently based on priority, wait time, and resource availability. A smaller job (1 node) will typically start before a larger one (2 nodes) regardless of submission order.

To chain jobs so that Job B only starts after Job A succeeds:

```bash
# Submit Job A, capture its job ID
JOB_A=$(sbatch job_a.sh | awk '{print $NF}')

# Submit Job B with a dependency on Job A
sbatch --dependency=afterok:$JOB_A job_b.sh
```

### Dependency types

| Flag | Meaning |
|---|---|
| `afterok:<jobid>` | Start only if the dependency completed with exit code 0 |
| `afternotok:<jobid>` | Start only if the dependency failed |
| `afterany:<jobid>` | Start after the dependency finishes, regardless of exit code |
| `after:<jobid>` | Start after the dependency has begun (not necessarily finished) |

### Chaining multiple jobs (e.g. pipeline)

```bash
JOB1=$(sbatch preprocess.sh | awk '{print $NF}')
JOB2=$(sbatch --dependency=afterok:$JOB1 train.sh | awk '{print $NF}')
JOB3=$(sbatch --dependency=afterok:$JOB2 evaluate.sh | awk '{print $NF}')
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
```

If a job in the chain fails, all downstream jobs with `afterok` dependencies are automatically cancelled.

---

## Quick Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch` | `common.sh` not sourced, or conda env not activated |
| Job stuck in `PD` > 30 min | Check `%R` reason; try `--partition=gpu_long` or reduce resources |
| Multi-node job hangs at barrier | Run `ip a` in interactive session, confirm `NCCL_SOCKET_IFNAME` matches actual interface |
| `CUDA_VISIBLE_DEVICES` mismatch | Ensure `--nproc_per_node` == `--gres=gpu:N` |
| `.err` file has `srun: error: PMK_KVS` | Switch `--rdzv_backend` to `env` or `c10d` |
