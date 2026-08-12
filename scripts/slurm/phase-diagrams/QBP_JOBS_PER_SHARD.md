# QBP_JOBS_PER_SHARD & Multi-Node Parallelization

Controls intra-shard parallelism: how many grid cells are computed concurrently within each SLURM task.

## Architecture

```
SLURM Allocation: 2 nodes × 8 tasks/node = 16 shards
        ↓
common_visualizer.sh (setup_visualizer_env)
        ↓
build cmd array + append_vqe_args + append_iqpe_args + append_dmrg_args
        ↓
run_visualizer_sharded_cmd(cmd)
        ↓
qbp_sharded.sh: spawn 16 parallel srun tasks
        ├─ shard 0: qbp run ${cmd} --task-index 0 --task-count 16
        ├─ shard 1: qbp run ${cmd} --task-index 1 --task-count 16
        ├─ ...
        └─ shard 15: qbp run ${cmd} --task-index 15 --task-count 16
        ↓
Aggregate results: qbp run ${cmd} --aggregate-only
```

## Execution Flow

1. **Script initialization** (`common_visualizer.sh`):
   - `setup_visualizer_env()` → sets LOG_DIR, PLOT_DIR, LOKY_DISABLE_RESOURCE_TRACKER=1, sources qbp_sharded.sh
   - `setup_visualizer_dmrg_env()` → sets JULIA_NUM_THREADS, OMP_NUM_THREADS, etc.
   - `SHARDS=16` explicitly set (2 nodes × 8 tasks/node)

2. **Command building**:
   - `append_vqe_args cmd` → VQE hyperparameters
   - `append_iqpe_args cmd` → IQPE hyperparameters + VQE-informed initial state
   - `append_dmrg_args cmd` → DMRG hyperparameters
   - `append_qbp_output_paths cmd` → log/plot paths

3. **Sharded execution** (`run_visualizer_sharded_cmd`):
   - Each of 16 shards gets a subset of the parameter grid
   - Parallel `srun` tasks with `2>/dev/null` (suppress loky warnings)
   - Each shard uses `QBP_JOBS_PER_SHARD` workers for repetition-level parallelism

4. **Aggregation**:
   - After all shards complete, aggregate results with `--aggregate-only`

## Method Parameters

All methods use sensible defaults from `common_visualizer.sh`:

| Method | VQE Iters | VQE Layers | VQE Reps | IQPE Time | IQPE Reps | DMRG Sweeps | DMRG Maxdims | Notes |
|--------|-----------|-----------|----------|-----------|-----------|-------------|--------------|-------|
| VQE | 10000 | 5 | 10 | — | — | — | — | CPU-bound |
| IQPE | 10000 | 5 | 10 | 0.75 | 20 | — | — | Uses VQE-informed initial state |
| DMRG | — | — | — | — | — | 4 | 20,50,100,200 | Memory-bound |
| Analytic | — | — | — | — | — | — | — | No parallelism |

### IQPE Initial State

All IQPE runs use:
- `--iqpe-initial-state vqe_informed`
- `--iqpe-initial-vqe-ansatz excitation_preserving`
- `--iqpe-initial-vqe-n-layers 2`
- `--iqpe-initial-vqe-ansatz-kwarg reps=2`
- `--iqpe-initial-vqe-max-iters 1000`

## Aggregation Strategy

- **IQPE**: Takes **median** of repetitions (reduces noise)
- **VQE**: Takes **minimum** of repetitions (finds best solution)
- **DMRG/Analytic**: Takes **first** (no variance)

## Usage

**Default behavior** (automatic):
```bash
sbatch scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/nocc_vs_t2.sh
```

Automatically:
- Uses 2 nodes × 8 tasks = 16 shards
- Distributes parameter grid across shards
- Each shard parallelizes repetitions with joblib

**Override QBP_JOBS_PER_SHARD if needed:**
```bash
# Reduce parallelism for memory constraints
export QBP_JOBS_PER_SHARD=2
sbatch scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/nocc_vs_t2.sh

# Use half the CPU budget
export QBP_JOBS_PER_SHARD=$((CPUS_PER_SHARD / 2))
sbatch scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/nocc_vs_t2.sh
```

**Interactive session:**
```bash
salloc -q interactive -C cpu -N 2 --ntasks-per-node=8 -c 16 -t 02:00:00 -A m5027
bash scripts/slurm/phase-diagrams/2x2/haldane/haldane_2x2_smoke_test.sh
```

## Intra-Shard Parallelism (QBP_JOBS_PER_SHARD)

`qbp/_run.py` calculates `jobs_per_shard()` automatically:
1. Check `QBP_JOBS_PER_SHARD` env var (explicit override)
2. Fall back to `os.sched_getaffinity(0)` (CPUs available, respects SLURM cgroups)
3. Fall back to `os.cpu_count()` (if affinity unavailable)

Used by `run_reps_parallel()` for repetition-level parallelism within each shard.

## Tuning

**Default auto-tuning:** `QBP_JOBS_PER_SHARD = CPUS_PER_SHARD`

Override only if:
- Node is memory-constrained (DMRG especially)
- Empirical testing shows reduced parallelism is faster (rare)

| Method | Auto-Tuned | Manual Min | Manual Max | Notes |
|--------|------------|-----------|-----------|-------|
| VQE | 16 (CPUS_PER_SHARD) | 2 | 8 | CPU-bound; ~4GB per job |
| IQPE | 16 (CPUS_PER_SHARD) | 2 | 8 | CPU-bound; ~4GB per job |
| DMRG | 16 (CPUS_PER_SHARD) | 1 | 4 | Memory-bound; reduce if OOM |

## Environment Variables

Set in `common_visualizer.sh`:
- `LOKY_DISABLE_RESOURCE_TRACKER=1` → suppress loky multiprocessing warnings
- `MPLBACKEND=Agg` → non-interactive matplotlib
- `JULIA_NUM_THREADS` → DMRG parallelism
- `OMP_NUM_THREADS=1` in qbp_sharded.sh (global OpenMP default)

## Output

All outputs go to `manuscript-plots/`:
- Logs: `manuscript-plots/logs/<model>/<lattice>/<sweep>/`
- Plots: `manuscript-plots/plots/<model>/<lattice>/<sweep>/`

## Examples

**Haldane 2×2 ground-state energy n_occ vs t2 (25-point smoke test):**
- Grid: 5 n_occ × 5 t2 = 25 cells
- Shards: 16 (most shards get ≤2 cells)
- Methods: iqpe, vqe, dmrg, analytic
- Reps per cell: VQE=10, IQPE=20
- Expected runtime: ~30-60 min (for smoke test with 25 cells)

**Haldane 2×2 ground-state energy n_occ vs t2 (production, 99 cells):**
- Grid: 9 n_occ × 11 t2 = 99 cells
- Shards: 16 (≈6 cells/shard)
- Methods: iqpe, vqe, dmrg, analytic
- Reps per cell: VQE=10, IQPE=20
- Expected runtime: ~6-12 hours (20 CPU-node-hours)
