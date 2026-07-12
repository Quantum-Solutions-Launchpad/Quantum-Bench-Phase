#!/bin/bash
#SBATCH -J haldane-2x2-smoke
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Smoke test: tiny n_occ-vs-t2 sweep for unified parallel runner.
# Verifies parallel methods execution, progress logging, and plots.
# Run interactively:
#   salloc -q interactive -C cpu -N 1 -t 01:00:00 -A m5027
#   bash scripts/slurm/phase-diagrams/2x2/haldane/haldane_2x2_smoke_test.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
SHARDS=16
setup_visualizer_env
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-8}"

export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

OUT_LOG_DIR="${LOG_DIR}/haldane/2x2/smoke-test"
OUT_PLOT_DIR="${PLOT_DIR}/haldane/2x2/smoke-test"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

# Reduced VQE/IQPE for smoke test (faster), matching backup
export VQE_ITERS="${VQE_ITERS:-100}"
export VQE_LAYERS="${VQE_LAYERS:-2}"
export VQE_REPS="${VQE_REPS:-1}"
export IQPE_TIME="${IQPE_TIME:-0.1}"
export IQPE_TROT="${IQPE_TROT:-1}"
export IQPE_ITERS="${IQPE_ITERS:-2}"
export IQPE_REPS="${IQPE_REPS:-1}"

echo "==================================================================="
echo "Haldane 2x2 SMOKE TEST (9-point grid)"
echo "  sweep:       n_occ [0 8 4] vs t2 [0.0 1.0 0.5]"
echo "  grid points: 3 × 3 = 9 combinations"
echo "  methods:     iqpe vqe dmrg analytic"
echo "  VQE:         iters=100, layers=2, reps=1"
echo "  IQPE:        time=0.1, trot=1, iters=2, reps=1"
echo "  DMRG:        nsweeps=4"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model haldane
    --method iqpe vqe dmrg analytic
    --lattice 2 2
    --observable E
    --x-param n_occ
    --x-range 0 8 4
    --y-param t2
    --y-range 0.0 1.0 0.5
    --t1 1.0
    --phi 0.7853981633974483
    --M 0.0
)

append_vqe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/smoke-test-all-E.json" "${OUT_PLOT_DIR}/smoke-test-all-E.pdf"

# Use basic IQPE (Hartree-Fock init, no VQE-informed) to match backup
cmd+=(
    --iqpe-time "${IQPE_TIME:-0.1}"
    --iqpe-trot "${IQPE_TROT:-1}"
    --iqpe-iters "${IQPE_ITERS:-2}"
    --iqpe-reps "${IQPE_REPS:-1}"
)
append_dmrg_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/smoke-test-all-E.json" "${OUT_PLOT_DIR}/smoke-test-all-E.pdf"

run_visualizer_sharded_cmd cmd
EXIT_STATUS=$?

echo ""
echo "==================================================================="
if (( EXIT_STATUS == 0 )); then
    echo "✓ SMOKE TEST PASSED at $(date)"
    echo ""
    echo "Data:  ${OUT_LOG_DIR}/smoke-test-all-E.json"
    echo "Plot:  ${OUT_PLOT_DIR}/smoke-test-all-E.pdf"
    ls -lh "${OUT_LOG_DIR}"/smoke-test-all-E.* 2>/dev/null | sed 's/^/  /'
    ls -lh "${OUT_PLOT_DIR}"/smoke-test-all-E.* 2>/dev/null | sed 's/^/  /'
else
    echo "✗ SMOKE TEST FAILED"
fi
echo "==================================================================="

exit ${EXIT_STATUS}
