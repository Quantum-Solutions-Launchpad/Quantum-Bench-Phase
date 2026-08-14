#!/bin/bash
#SBATCH -J hubbard-2x2-mag-vqe-m-stag
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Hubbard 2x2 staggered magnetization: analytic reference + VQE heatmap.

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

OUT_LOG_DIR="${LOG_DIR}/hubbard/2x2"
OUT_PLOT_DIR="${PLOT_DIR}/hubbard/2x2"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

OBSERVABLE="M_stag"
SWEEP_TAG="n_occ-vs-U"

echo "==================================================================="
echo "Hubbard 2x2 magnetization analytic+VQE (SHARDED) - M_stag"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-16} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-10.0} ${HUBBARD_U_STEP:-0.5}]"
echo "  methods:     analytic vqe"
echo "  optimizer:   L_BFGS_B"
echo "  plot format: heatmap"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model hubbard
    --method analytic vqe
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --heatmap
    --x-param n_occ
    --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-16}" "${HUBBARD_N_OCC_STEP:-1}"
    --y-param U
    --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-10.0}" "${HUBBARD_U_STEP:-0.5}"
    --t "${HUBBARD_T:-1.0}"
    --vqe-warm-start
    --vqe-layers "${VQE_LAYERS:-4}"
    --vqe-reps "${VQE_REPS:-5}"
    --vqe-optimizer "${VQE_OPTIMIZER:-L_BFGS_B}"
    --vqe-iters "${VQE_ITERS:-400}"
)

append_qbp_output_paths cmd "${OUT_LOG_DIR}/vqe-${OBSERVABLE}-heatmap-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/vqe-${OBSERVABLE}-heatmap-${SWEEP_TAG}.pdf"

echo ""
echo "==================================================================="
if run_visualizer_sharded_cmd cmd; then
    echo "✓ JOB PASSED at $(date)"
else
    EXIT_STATUS=$?
    echo "✗ JOB FAILED"
    echo "status: ${EXIT_STATUS}"
    echo "==================================================================="
    exit "${EXIT_STATUS}"
fi
echo "==================================================================="
