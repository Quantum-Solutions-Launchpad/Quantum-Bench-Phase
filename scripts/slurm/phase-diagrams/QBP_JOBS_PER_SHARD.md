# QBP_JOBS_PER_SHARD

Controls intra-shard parallelism: how many grid cells are computed concurrently within each SLURM task.

## Architecture

```
SLURM: N shards × M cells/shard (via --task-index / --task-count)
Each shard runs K workers in parallel (QBP_JOBS_PER_SHARD)
Workers computed rep results via joblib.Parallel
```

## Implementation

`qbp/_run.py` calculates `jobs_per_shard()` automatically:
- Checks `QBP_JOBS_PER_SHARD` env var first (explicit override)
- Falls back to `os.sched_getaffinity(0)` — CPUs available to this process (respects SLURM cgroup limits)
- Falls back to `os.cpu_count()` if affinity detection unavailable
- Used by `run_reps_parallel()` for repetition-level parallelism

## Methods

| Method | WANTS_PARALLEL | Cell Cost |
|--------|---|---|
| VQE | True | 10-100s |
| IQPE | True | 5-50s |
| DMRG | True | 30-300s |
| Analytic | False | ~1ms |

## Usage

**Default behavior** (no configuration needed):
```bash
# Auto-detects CPUS_PER_SHARD and uses that as parallelism
sbatch scripts/slurm/phase-diagrams/2x2/hubbard/magnetization/vqe.sh
```

**Override if needed:**
```bash
# Reduce parallelism for memory-constrained nodes
export QBP_JOBS_PER_SHARD=2
sbatch scripts/slurm/phase-diagrams/2x2/hubbard/magnetization/vqe.sh

# Custom value (e.g., half the allocated CPUs for I/O headroom)
export QBP_JOBS_PER_SHARD=$((SLURM_CPUS_PER_TASK / 2))
sbatch scripts/slurm/phase-diagrams/2x2/hubbard/magnetization/dmrg.sh
```

## Tuning

**Default auto-tuning** now sets `QBP_JOBS_PER_SHARD = CPUS_PER_SHARD`. Override only if:
- Node is memory-constrained
- You want to reserve CPUs for other tasks
- Empirical testing shows reduced parallelism is faster (rare)

| Method | Auto-tuned | Manual Override | Notes |
|--------|---|---|---|
| VQE | `CPUS_PER_SHARD` | 2-4 | CPU-bound; memory ~4GB/job |
| IQPE | `CPUS_PER_SHARD` | 2-4 | CPU-bound; memory ~4GB/job |
| DMRG | `CPUS_PER_SHARD` | 1-2 | Memory-bound; reduce if OOM |

## Examples

**Hubbard 2×2 (DMRG, 357 cells, 32 shards ≈ 11 cells/shard):**
- QBP_JOBS_PER_SHARD=1: ~20h serial
- QBP_JOBS_PER_SHARD=2: ~12h (50% speedup)
- QBP_JOBS_PER_SHARD=4: ~7h (overhead caps gains)

**Hubbard 3×3 (VQE, 36 qubits):**
- 2 recommended (4 jobs × 4GB = 16GB per shard)
- Monitor memory; OOM risk above 2
