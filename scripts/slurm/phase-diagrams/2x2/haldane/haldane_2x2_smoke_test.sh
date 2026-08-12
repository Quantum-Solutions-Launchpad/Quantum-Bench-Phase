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

# Smoke test: Haldane 2x2 VQE + IQPE (minimal sweep for pipeline verification).
# Based on: nocc_vs_t2_hard_wall_vqe_iqpe.sh but with reduced parameters.
# Verifies VQE + IQPE execution, progress logging, and plots.
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
echo "Haldane 2x2 VQE + IQPE SMOKE TEST (6-point grid)"
echo "  sweep:       n_occ [0 4 2] vs t2 [0.0 1.0 1.0]"
echo "  grid points: 3 × 2 = 6 combinations"
echo "  boundary:    open (hard wall)"
echo "  methods:     vqe iqpe"
echo "  VQE:         iters=100, layers=2, reps=1"
echo "  IQPE:        time=0.1, trot=1, iters=2, reps=1"
echo "  parallelism: distributed across ${SHARDS} shards on single node"
echo "==================================================================="

cmd=(
    --model haldane
    --method vqe iqpe
    --lattice 2 2
    --observable E
    --boundary open
    --x-param n_occ
    --x-range 0 4 2
    --y-param t2
    --y-range 0.0 1.0 1.0
    --t1 1.0
    --phi 0.7853981633974483
    --M 0.0
)

append_vqe_args cmd
append_iqpe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/smoke-test-vqe-iqpe-E.json" "${OUT_PLOT_DIR}/smoke-test-vqe-iqpe-E.pdf"

echo ""
echo "==================================================================="
if run_visualizer_sharded_cmd cmd; then
    echo "✓ SMOKE TEST PASSED at $(date)"
    echo ""
    echo "Data:  ${OUT_LOG_DIR}/smoke-test-vqe-iqpe-E.json"
    echo "Plot:  ${OUT_PLOT_DIR}/smoke-test-vqe-iqpe-E.pdf"
    ls -lh "${OUT_LOG_DIR}"/smoke-test-vqe-iqpe-E.* 2>/dev/null | sed 's/^/  /'
    ls -lh "${OUT_PLOT_DIR}"/smoke-test-vqe-iqpe-E.* 2>/dev/null | sed 's/^/  /'
else
    EXIT_STATUS=$?
    echo "✗ SMOKE TEST FAILED"
    echo "status: ${EXIT_STATUS}"
    echo "==================================================================="
    exit "${EXIT_STATUS}"
fi
echo "==================================================================="
