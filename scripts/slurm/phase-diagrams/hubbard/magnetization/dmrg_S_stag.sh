#!/bin/bash
#SBATCH -J hubbard-2x1-mag-dmrg-s-stag
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Hubbard 2x1 structure factor (staggered): DMRG heatmap.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
SHARDS=4
setup_visualizer_env
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-8}"

export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

OUT_LOG_DIR="${LOG_DIR}/hubbard/2x1/magnetization"
OUT_PLOT_DIR="${PLOT_DIR}/hubbard/2x1/magnetization"
OUT_EVAL_DIR="/pscratch/sd/m/mbao202/NNL-P7/evaluation/hubbard/2x1/magnetization"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}" "${OUT_EVAL_DIR}"

OBSERVABLE="S_stag"
SWEEP_TAG="n_occ-vs-U"

echo "==================================================================="
echo "Hubbard 2x1 structure factor DMRG (SHARDED) - S_stag"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-8} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-80.0} ${HUBBARD_U_STEP:-1.0}]"
echo "  method:      dmrg"
echo "  plot format: heatmap"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model hubbard-honeycomb
    --method vqe
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --heatmap
    --x-param n_occ
    --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-8}" "${HUBBARD_N_OCC_STEP:-1}"
    --y-param U
    --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-80.0}" "${HUBBARD_U_STEP:-1.0}"
    --t "${HUBBARD_T:-1.0}"
    --no-dmrg-conserve-sz
    --dmrg-initial-state "${DMRG_INITIAL_STATE:-neel}"
)

append_dmrg_args cmd
append_qbp_output_paths cmd "${OUT_EVAL_DIR}/hubbard-2x1-ideal-dmrg-${OBSERVABLE}-heatmap-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-ideal-dmrg-${OBSERVABLE}-heatmap-${SWEEP_TAG}.pdf"

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
