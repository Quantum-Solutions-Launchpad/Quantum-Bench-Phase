#!/bin/bash
#SBATCH -J hubbard-2x2-mag
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --array=0-1
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.err

# Unified parallel runner: Hubbard 2x2 magnetization n_occ vs U sweep.
# Runs all methods (analytic, vqe, dmrg) in a SINGLE qbp run call
# with joblib.Parallel() scheduling ALL jobs simultaneously.
# Uses SLURM array tasks for M_stag (0) and M_total (1).
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

OBSERVABLES=(M_stag M_total)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID < 0 || TASK_ID >= ${#OBSERVABLES[@]} )); then
    echo "TASK_ID must be 0 (M_stag) or 1 (M_total)." >&2
    exit 1
fi
OBSERVABLE="${OBSERVABLE:-${OBSERVABLES[$TASK_ID]}}"
SWEEP_TAG="${SWEEP_TAG:-n_occ-vs-U}"

PIPELINE="${PIPELINE:-ideal}"
backend_args=()
if [[ "${PIPELINE}" == "noisy" ]]; then
    backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
fi

HEATMAP="${HEATMAP:-1}"

echo "==================================================================="
echo "Hubbard 2x2 magnetization (SHARDED)"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-16} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-10.0} ${HUBBARD_U_STEP:-0.5}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  methods:     analytic vqe dmrg"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "  plot format: $([[ "${HEATMAP}" == "1" ]] && echo heatmap || echo 3d)"
echo "==================================================================="

cmd=(
    --model hubbard
    --method analytic vqe dmrg
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --x-param n_occ
    --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-16}" "${HUBBARD_N_OCC_STEP:-1}"
    --y-param U
    --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-10.0}" "${HUBBARD_U_STEP:-0.5}"
    --t "${HUBBARD_T:-1.0}"
    --dmrg-nsweeps "${DMRG_NSWEEPS:-4}"
    --dmrg-maxdims "${DMRG_MAXDIMS:-20,50,100,200}"
    --dmrg-cutoff "${DMRG_CUTOFF:-1e-9}"
    --dmrg-seed "${DMRG_SEED:-1234}"
    --no-dmrg-conserve-sz
    --dmrg-initial-state "${DMRG_INITIAL_STATE:-neel}"
)

if [[ "${HEATMAP}" == "1" ]]; then
    cmd+=(--heatmap)
fi

if [[ "${PIPELINE}" == "noisy" ]]; then
    cmd+=("${backend_args[@]}")
fi

append_vqe_args cmd
append_dmrg_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-${PIPELINE}-all-${OBSERVABLE}-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-all-${OBSERVABLE}-${SWEEP_TAG}.pdf"

if run_visualizer_sharded_cmd cmd; then
    echo "Completed at $(date)"
else
    EXIT_STATUS=$?
    echo "ERROR: Computation failed with status ${EXIT_STATUS}"
    exit "${EXIT_STATUS}"
fi
