#!/bin/bash
#SBATCH -J haldane-2x2-smoke
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Smoke test: tiny n_occ-vs-t2 sweep through the shared ground_state_energy
# driver. Verifies sharding, parallel methods, progress logging, and plots.
# Run interactively:
#   salloc -q interactive -C cpu -N 1 -t 01:00:00 -A m5027
#   bash scripts/slurm/phase-diagrams/2x2/haldane/haldane_2x2_smoke_test.sh

set -euo pipefail

# Same sweep as nocc_vs_t2.sh, just a tiny 3x3 grid
export X_PARAM=n_occ
export X_START=0 X_END=2 X_STEP=1
export Y_PARAM=t2
export Y_START=0.0 Y_END=1.0 Y_STEP=0.5
export SWEEP_TAG=smoke-test-nocc-vs-t2  # keep outputs separate from real runs
export OUT_SUBDIR=smoke-test            # nest outputs under haldane/2x2/smoke-test/

# Use regular job hyperparameters for realistic testing
export VQE_ITERS=10 VQE_LAYERS=5 VQE_REPS=1
export IQPE_TIME=0.2 IQPE_TROT=5 IQPE_ITERS=8 IQPE_REPS=1
export DMRG_NSWEEPS=4 DMRG_MAXDIMS="20,50,100,200"

# Keep resource use small enough for a single interactive node
export SHARDS="${SHARDS:-4}"
export CPUS_PER_SHARD="${CPUS_PER_SHARD:-8}"
export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
# common.sh verifies all outputs (summaries, plots, journals)
# and exits non-zero if anything is missing.
if bash "${REPO_ROOT}/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/common.sh"; then
    echo "SMOKE TEST PASSED at $(date)"
else
    echo "SMOKE TEST FAILED" >&2
    exit 1
fi
