#!/bin/bash
#SBATCH -J hubbard-2x2-mag-analytic
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --array=0-1
#SBATCH -t 02:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.err

# Hubbard 2x2 magnetization ANALYTIC: n_occ vs U sweep for M_stag/M_total.
# Uses unified parallel runner for optimal scheduling (even for single method).
# Array tasks: 0=M_stag, 1=M_total.
# Much faster than re-rendering with analytic_plots.sh.

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

OBSERVABLES=(M_stag M_total)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID < 0 || TASK_ID >= ${#OBSERVABLES[@]} )); then
    echo "TASK_ID must be 0 (M_stag) or 1 (M_total)." >&2
    exit 1
fi
OBSERVABLE="${OBSERVABLE:-${OBSERVABLES[$TASK_ID]}}"
SWEEP_TAG="n_occ-vs-U"

echo "==================================================================="
echo "Hubbard 2x2 magnetization ANALYTIC (SHARDED)"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-16} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-10.0} ${HUBBARD_U_STEP:-0.5}]"
echo "  method:      analytic (exact diagonalization)"
echo "  plot format: heatmap"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model hubbard
    --method analytic
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --x-param n_occ
    --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-16}" "${HUBBARD_N_OCC_STEP:-1}"
    --y-param U
    --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-10.0}" "${HUBBARD_U_STEP:-0.5}"
    --t "${HUBBARD_T:-1.0}"
    --heatmap
)

append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-ideal-analytic-${OBSERVABLE}-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-ideal-analytic-${OBSERVABLE}-${SWEEP_TAG}.pdf"

echo ""
echo "==================================================================="
if run_visualizer_sharded_cmd cmd; then
    echo "✓ Completed at $(date)"
    echo "  output: ${OUT_LOG_DIR}/simulated-ideal-analytic-${OBSERVABLE}-${SWEEP_TAG}.json"
    echo "  plot:   ${OUT_PLOT_DIR}/simulated-ideal-analytic-${OBSERVABLE}-${SWEEP_TAG}.pdf"
else
    EXIT_STATUS=$?
    echo "✗ FAILED with status ${EXIT_STATUS}"
    echo "==================================================================="
    exit "${EXIT_STATUS}"
fi
echo "==================================================================="
