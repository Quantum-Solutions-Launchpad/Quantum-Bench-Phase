#!/bin/bash
#SBATCH -J hubbard-2x2-smoke
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Smoke test: tiny n_occ-vs-U sweep for VQE magnetization (M_stag/M_total).
# Verifies VQE execution, array job scheduling, progress logging, and plots.
# Run interactively:
#   salloc -q interactive -C cpu -N 2 -t 01:00:00 -A m5027
#   bash scripts/slurm/phase-diagrams/2x2/hubbard/hubbard_2x2_smoke_test.sh

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

OUT_LOG_DIR="${LOG_DIR}/hubbard/2x2/smoke-test"
OUT_PLOT_DIR="${PLOT_DIR}/hubbard/2x2/smoke-test"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

# Reduced VQE for smoke test (faster)
export VQE_ITERS="${VQE_ITERS:-100}"
export VQE_LAYERS="${VQE_LAYERS:-2}"
export VQE_REPS="${VQE_REPS:-1}"

# Test M_stag observable (array task 0), or M_total (array task 1)
OBSERVABLE="${OBSERVABLE:-M_stag}"
SWEEP_TAG="smoke-test-nocc-vs-U"

echo "==================================================================="
echo "Hubbard 2x2 SMOKE TEST - VQE Magnetization (tiny 2x3 grid)"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [0 2 1] vs U [0.0 1.0 0.5]"
echo "  method:      vqe"
echo "  VQE:         iters=100, layers=2, reps=1"
echo "  parallelism: distributed across ${SHARDS} shards across nodes"
echo "==================================================================="

cmd=(
    --model hubbard-honeycomb
    --method vqe
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --x-param n_occ
    --x-range 0 2 1
    --y-param U
    --y-range 0.0 1.0 0.5
    --t 1.0
    --heatmap
)

append_vqe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/smoke-test-${OBSERVABLE}.json" "${OUT_PLOT_DIR}/smoke-test-${OBSERVABLE}.pdf"

echo ""
echo "==================================================================="
if run_visualizer_sharded_cmd cmd; then
    echo "✓ SMOKE TEST PASSED at $(date)"
    echo ""
    echo "Data:  ${OUT_LOG_DIR}/smoke-test-${OBSERVABLE}.json"
    echo "Plot:  ${OUT_PLOT_DIR}/smoke-test-${OBSERVABLE}.pdf"
    ls -lh "${OUT_LOG_DIR}"/smoke-test-${OBSERVABLE}.* 2>/dev/null | sed 's/^/  /'
    ls -lh "${OUT_PLOT_DIR}"/smoke-test-${OBSERVABLE}.* 2>/dev/null | sed 's/^/  /'
else
    EXIT_STATUS=$?
    echo "✗ SMOKE TEST FAILED"
    echo "status: ${EXIT_STATUS}"
    echo "==================================================================="
    exit "${EXIT_STATUS}"
fi
echo "==================================================================="
