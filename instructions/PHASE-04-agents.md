# PHASE 04: Sophisticated Multi-Agent System

**Branch**: `feature/04-agentic-system`  
**Prerequisite**: `feature/03-fsdp-hpc-sharding` (must have a trained FSDP checkpoint).  
**Goal**: Build a ReAct-style multi-agent system (Planner, Executor, Critic) that uses your Phase 3 model as the brain, runs across Amarel nodes, and performs tool-based tasks on the cluster's filesystem.

---

## 1. Objective
Load the sharded FSDP checkpoint for inference. Implement a ReAct loop manually (no LangChain). Partition agents across Amarel: GPU node for Planner/Critic, CPU nodes for Executors. Connect them via a shared filesystem or Redis for inter-process communication (IPC).

---

## 2. Files to Create / Modify
- `src/inference_engine.py` – Loads FSDP checkpoint in eval mode, handles tokenization, generation with KV-caching.
- `src/agents/__init__.py`
- `src/agents/base_agent.py` – Abstract class with `act(observation)` and `reflect()` methods.
- `src/agents/planner.py` – ReAct loop: Thought → Action → Observation. Calls the model via `inference_engine`.
- `src/agents/executor.py` – Runs tools (Python REPL, file read/write) on CPU-only nodes.
- `src/agents/critic.py` – Validates executor outputs, suggests corrections, decides when task is complete.
- `src/agents/memory.py` – Shared memory using SQLite or file-based locks (fcntl) for Amarel's shared `$SCRATCH`.
- `tools/__init__.py`
- `tools/python_repl.py` – Executes safe Python code with restricted globals.
- `tools/file_reader.py` – Reads/writes files under `$SCRATCH` with path validation (prevent directory traversal).
- `configs/phase4_agent_config.yaml` – Model path, tool whitelist, max ReAct steps, Redis/DB connection settings.
- `orchestrator.py` – Entry point: spawns Planner, dispatches tasks to Executors, loops until completion.
- `slurm/launch_planner.sh` – `#SBATCH --gres=gpu:1` (runs Planner + Critic on the same GPU).
- `slurm/launch_executors.sh` – `#SBATCH --nodes=2 --ntasks-per-node=1 --cpus-per-task=8` (CPU-only for Executors).

---

## 3. Acceptance Criteria
- [ ] `inference_engine.py` loads the Phase 3 sharded checkpoint and generates coherent text with KV-caching.
- [ ] Planner, Executor, and Critic communicate via shared files (JSONL messages) or Redis without corruption.
- [ ] Agent successfully completes a task: "Read all `.txt` files in `$SCRATCH/data/`, summarize each, and write a combined report to `$SCRATCH/output/report.md`."
- [ ] The ReAct loop terminates gracefully (max steps = 10) and logs the full decision trace.
- [ ] Executors run on CPU-only nodes (no GPUs allocated) to save cluster resources.

---

## 4. Amarel Edge Case (Must Handle)
- **File Locking**: Use `fcntl.flock` on shared JSON message files to prevent race conditions when multiple executors write results.
- **IPC Backbone**: If Redis is unavailable, fall back to SQLite with WAL mode or a simple directory-based queue (`$SCRATCH/agent_queue/`).
- **Model Loading**: Loading a full 1B FSDP checkpoint on a single GPU for inference may require `torch.distributed` even for a single rank—use `cpu_offload` or shard-to-cpu if memory is tight.
- **SLURM Dependencies**: Launch Planner first, then Executors, or use `--dependency=afterok` in SLURM to ensure agents start in order.

---

## 5. Cursor Instructions
- Follow `@GLOBAL.md`: no LangChain or third-party agent frameworks—implement ReAct from scratch.
- Use `torch.no_grad()` for all inference. Use `temperature=0.7` and `top_p=0.9` for diversity.
- Keep the agent state (conversation history) in a Python `deque` with max length = 20 to avoid context overflow.
- Log every Thought/Action/Observation to `$SCRATCH/agent_logs/` for debugging and portfolio traces.