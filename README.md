# Amarel LLM Foundry

**5-phase journey**: From SLURM hello-world → pure PyTorch transformer → DDP → FSDP multi-node → multi-agent system.

## Project Map
| Phase | Branch | Goal |
| :--- | :--- | :--- |
| 0 | `feature/00-hpc-distributed-baseline` | Validate `torch.distributed` across Amarel nodes |
| 1 | `feature/01-nanogpt-transformer` | 124M GPT from scratch, single-GPU |
| 2 | `feature/02-ddp-multi-gpu` | DDP scaling to 4 GPUs |
| 3 | `feature/03-fsdp-hpc-sharding` | FSDP multi-node sharding (1B params) |
| 4 | `feature/04-agentic-system` | ReAct agents with trained checkpoint |

## Getting Started
1. Clone and set up environment:
   ```bash
   pip install -e .
   ```

2. Source SLURM config:
    ```bash
    source slurm/common.sh
    ```

3. Switch to a feature branch and follow `instructions/PHASE-XX.md`.

## Repository Structure
- `instructions/` – Cursor agent blueprints for each phase.
- `slurm/` – SLURM scripts and shared environment.
- `configs/` – YAML hyperparameter configs.
- `docs/` – Research notes.
- `src/` – Code (populated in feature branches).
