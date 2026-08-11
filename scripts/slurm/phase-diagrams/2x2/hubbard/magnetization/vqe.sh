#!/bin/bash
# Hubbard 2x2 magnetization VQE: n_occ vs U sweep for M_stag/M_total.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_vqe_env

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
echo "Hubbard 2x2 magnetization VQE"
echo "  observable:  ${OBSERVABLE}"
echo "  sweep:       n_occ [0-16] vs U [0-10]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  method:      vqe"
echo "==================================================================="

cmd=(
    --model hubbard
    --method vqe
    --lattice 2 2
    --observable "${OBSERVABLE}"
    --x-param n_occ
    --x-range 0 16 1
    --y-param U
    --y-range 0.0 10.0 0.5
    --t 1.0
    --vqe-ansatz uccsd
    --vqe-optimizer cobyla
)

if [[ "${PIPELINE}" == "noisy" ]]; then
    cmd+=("${backend_args[@]}")
fi

append_qbp_output_paths cmd "${OUT_EVAL_DIR}/hubbard-2x2-${PIPELINE}-vqe-${OBSERVABLE}-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-vqe-${OBSERVABLE}-${SWEEP_TAG}.pdf"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Running in interactive mode"
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
