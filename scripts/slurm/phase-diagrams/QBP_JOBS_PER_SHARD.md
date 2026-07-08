# QBP_JOBS_PER_SHARD

Controls intra-shard parallelism: how many grid cells are computed concurrently within each SLURM task.

## Architecture

```
SLURM: N shards × M cells/shard (via --task-index / --task-count)
Each shard runs K workers in parallel (QBP_JOBS_PER_SHARD)
Workers computed rep results via joblib.Parallel
```

## Implementation

`_core._rep_jobs()` calculates worker count:
- `QBP_REP_JOBS` env var overrides all
- Otherwise: `SLURM_CPUS_PER_TASK / QBP_JOBS_PER_SHARD`
- Used by `run_reps_parallel()` for repetition-level parallelism

## Methods

| Method | WANTS_PARALLEL | Cell Cost |
|--------|---|---|
| VQE | True | 10-100s |
| IQPE | True | 5-50s |
| DMRG | True | 30-300s |
| Analytic | False | ~1ms |

## Usage

```bash
# Runtime override
export QBP_JOBS_PER_SHARD=2
sbatch scripts/slurm/phase-diagrams/2x2/hubbard/magnetization/vqe.sh

# Adaptive: use half CPUs per shard for I/O headroom
export QBP_JOBS_PER_SHARD=$((SLURM_CPUS_PER_TASK / 2))
```

## Tuning

| Method | Recommended | Notes |
|--------|---|---|
| VQE | 2-4 | CPU-bound; memory ~4GB/job |
| IQPE | 2-4 | CPU-bound; memory ~4GB/job |
| DMRG | 1-2 | Memory-bound; check node capacity |

## Examples

**Hubbard 2×2 (DMRG, 357 cells, 32 shards ≈ 11 cells/shard):**
- QBP_JOBS_PER_SHARD=1: ~20h serial
- QBP_JOBS_PER_SHARD=2: ~12h (50% speedup)
- QBP_JOBS_PER_SHARD=4: ~7h (overhead caps gains)

**Hubbard 3×3 (VQE, 36 qubits):**
- 2 recommended (4 jobs × 4GB = 16GB per shard)
- Monitor memory; OOM risk above 2
