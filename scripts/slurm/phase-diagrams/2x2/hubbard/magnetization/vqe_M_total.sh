#!/bin/bash
#SBATCH -J hubbard-2x2-mag-vqe-m-total
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Hubbard 2x2 magnetization VQE: n_occ vs U sweep for M_total.
# Uses unified parallel runner for optimal scheduling.
# PIPELINE=noisy sbatch ... for the noisy pipeline (default: ideal).

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

OUT_LOG_DIR="${LOG_DIR}/hubbard/2x2/magnetization"
OUT_PLOT_DIR="${PLOT_DIR}/hubbard/2x2/magnetization"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

OBSERVABLE="M_total"
SWEEP_TAG="n_occ-vs-U"

PIPELINE="${PIPELINE:-ideal}"
backend_args=()
if [[ "${PIPELINE}" == "noisy" ]]; then
    backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
fi

echo "==================================================================="
echo "Hubbard 2x2 magnetization VQE (SHARDED) - M_total"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-16} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-10.0} ${HUBBARD_U_STEP:-0.5}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  method:      vqe"
echo "  plot format: heatmap"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model hubbard
    --method vqe
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --x-param n_occ
    --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-16}" "${HUBBARD_N_OCC_STEP:-1}"
    --y-param U
    --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-10.0}" "${HUBBARD_U_STEP:-0.5}"
    --t "${HUBBARD_T:-1.0}"
    --heatmap
)

if [[ "${PIPELINE}" == "noisy" ]]; then
    cmd+=("${backend_args[@]}")
fi

append_vqe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-${PIPELINE}-vqe-${OBSERVABLE}-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-vqe-${OBSERVABLE}-${SWEEP_TAG}.pdf"

run_visualizer_sharded_cmd cmd
EXIT_STATUS=$?

echo ""
echo "==================================================================="
if (( EXIT_STATUS == 0 )); then
    echo "✓ JOB PASSED at $(date)"
else
    echo "✗ JOB FAILED"
fi
echo "==================================================================="

exit ${EXIT_STATUS}
