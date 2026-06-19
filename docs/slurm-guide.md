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
| `NCCL_ASYNC_ERROR_HANDLING` | `1` | Surfaces clear errors instead of hanging |
| `NCCL_IB_DISABLE` | `1` | Amarel uses Ethernet, not InfiniBand |
| `NCCL_SOCKET_IFNAME` | `eth0` | Force NCCL to use the correct NIC (verify with `ip a` on compute node if multi-node hangs) |
| `NCCL_TIMEOUT` | `1800` | 30-min timeout for slow cross-node init |

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
