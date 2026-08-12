#!/bin/bash
# Hubbard 2x2 magnetization DMRG: n_occ vs U sweep for M_stag/M_total.
# Uses unified parallel runner for optimal scheduling (even for single method).
# For interactive session: TASK_ID=0 bash dmrg.sh (or TASK_ID=1 for M_total)
# For batch: sbatch dmrg.sh
# PIPELINE=noisy TASK_ID=0 bash dmrg.sh for noisy pipeline (default: ideal).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
# For interactive sessions: use fewer shards for faster execution
SHARDS="${SHARDS:-4}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
setup_visualizer_env
setup_visualizer_dmrg_env "${CPUS_PER_TASK}"

export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

OUT_LOG_DIR="${LOG_DIR}/hubbard/2x2/magnetization"
OUT_PLOT_DIR="${PLOT_DIR}/hubbard/2x2/magnetization"
OUT_EVAL_DIR="/pscratch/sd/m/mbao202/NNL-P7/evaluation/hubbard/2x2/magnetization"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}" "${OUT_EVAL_DIR}"

OBSERVABLES=(M_stag M_total)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID < 0 || TASK_ID >= ${#OBSERVABLES[@]} )); then
    echo "TASK_ID must be 0 (M_stag) or 1 (M_total)." >&2
    exit 1
fi
OBSERVABLE="${OBSERVABLE:-${OBSERVABLES[$TASK_ID]}}"
SWEEP_TAG="n_occ-vs-U"

PIPELINE="${PIPELINE:-ideal}"
backend_args=()
if [[ "${PIPELINE}" == "noisy" ]]; then
    backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
fi

echo "==================================================================="
echo "Hubbard 2x2 magnetization DMRG (SHARDED)"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-16} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-10.0} ${HUBBARD_U_STEP:-0.5}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  method:      dmrg"
echo "  plot format: heatmap"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model hubbard
    --method dmrg
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --x-param n_occ
    --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-16}" "${HUBBARD_N_OCC_STEP:-1}"
    --y-param U
    --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-10.0}" "${HUBBARD_U_STEP:-0.5}"
    --t "${HUBBARD_T:-1.0}"
    --no-dmrg-conserve-sz
    --dmrg-initial-state "${DMRG_INITIAL_STATE:-neel}"
    --heatmap
)

if [[ "${PIPELINE}" == "noisy" ]]; then
    cmd+=("${backend_args[@]}")
fi

append_dmrg_args cmd
append_qbp_output_paths cmd "${OUT_EVAL_DIR}/hubbard-2x2-${PIPELINE}-dmrg-${OBSERVABLE}-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-dmrg-${OBSERVABLE}-${SWEEP_TAG}.pdf"

# For interactive sessions: run without sharding
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Running in interactive mode (no sharding)"
    if ${QBP_CLI} run "${cmd[@]}"; then
        echo "Completed at $(date)"
    else
        EXIT_STATUS=$?
        echo "ERROR: Computation failed with status ${EXIT_STATUS}"
        exit "${EXIT_STATUS}"
    fi
else
    if run_visualizer_sharded_cmd cmd; then
        echo "Completed at $(date)"
    else
        EXIT_STATUS=$?
        echo "ERROR: Computation failed with status ${EXIT_STATUS}"
        exit "${EXIT_STATUS}"
    fi
fi
