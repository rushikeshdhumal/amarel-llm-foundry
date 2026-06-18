# PHASE 00: HPC Distributed Baseline

**Branch**: `feature/00-hpc-distributed-baseline`  
**Prerequisite**: None. Start from `main`.  
**Goal**: Validate that you can launch PyTorch across Amarel nodes and understand SLURM environment variables.

---

## 1. Objective
Write a "Hello World" distributed script that prints the rank, world size, and hostname across multiple GPUs and nodes. This confirms that `torch.distributed` and NCCL are correctly configured on Amarel.

---

## 2. Files to Create
- `src/__init__.py` (empty)
- `src/dist_hello.py` – Script that initializes the process group and prints rank info.
- `slurm/hello_1node_1gpu.sh`
- `slurm/hello_1node_4gpu.sh`
- `slurm/hello_2nodes_4gpu.sh` (tests cross-node communication)

---

## 3. Acceptance Criteria
- [ ] `hello_1node_1gpu.sh` runs and prints `[Rank 0/1] Host: <hostname>`.
- [ ] `hello_1node_4gpu.sh` prints ranks 0,1,2,3 all on the same node.
- [ ] `hello_2nodes_4gpu.sh` prints ranks distributed across 2 nodes (hostnames differ).
- [ ] All scripts use `torchrun` (not `mpirun`) and source `slurm/common.sh`.

---

## 4. Amarel Edge Case (Must Handle)
- NCCL requires `MASTER_ADDR` and `MASTER_PORT`. Use `os.environ['SLURM_NODELIST']` to resolve the master node IP via `socket.gethostbyname()`.
- Set `os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"` to get clear errors if nodes fail to see each other.

---

## 5. Cursor Instructions
- Follow `@GLOBAL.md` rules: no hard-coded paths, use `$SCRATCH` only for logs (no heavy data yet).
- Use `argparse` to accept `--backend` (default `nccl` for GPU, `gloo` for fallback).
- Log output to `$SLURM_SUBMIT_DIR/hello_%j.out` via SLURM's native output redirection.