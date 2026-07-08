#!/bin/bash
#SBATCH -J hubbard-2x2-smoke
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Smoke test: tiny n_occ-vs-U sweep through the shared magnetization driver.
# Verifies sharding, parallel methods, progress logging, and plots.
# Run interactively:
#   salloc -q interactive -C cpu -N 1 -t 01:00:00 -A m5027
#   bash scripts/slurm/phase-diagrams/2x2/hubbard/hubbard_2x2_smoke_test.sh

set -euo pipefail

export TASK_ID="${TASK_ID:-0}"              # default to M_stag; TASK_ID=1 tests M_total
export SWEEP_TAG=smoke-test-nocc-vs-U       # keep outputs separate from real runs
export OUT_SUBDIR=smoke-test                # nest outputs under hubbard/2x2/smoke-test/

# Tiny 3x3 grid
export HUBBARD_N_OCC_START=0
export HUBBARD_N_OCC_END=2
export HUBBARD_N_OCC_STEP=1
export HUBBARD_U_START=0.0
export HUBBARD_U_END=1.0
export HUBBARD_U_STEP=0.5

# Minimal method parameters -- fast, not accurate
export METHODS="${METHODS:-analytic vqe dmrg}"
export VQE_ITERS=5 VQE_LAYERS=1 VQE_REPS=2
export IQPE_REPS=0
export DMRG_NSWEEPS=2 DMRG_MAXDIMS="10,20"

# Keep resource use small enough for a single interactive node
export SHARDS="${SHARDS:-4}"
export CPUS_PER_SHARD="${CPUS_PER_SHARD:-8}"
export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
# common.sh verifies all outputs (summaries, plots, journals)
# and exits non-zero if anything is missing.
if bash "${REPO_ROOT}/scripts/slurm/phase-diagrams/2x2/hubbard/magnetization/common.sh"; then
    echo "SMOKE TEST PASSED at $(date)"
else
    echo "SMOKE TEST FAILED" >&2
    exit 1
fi
