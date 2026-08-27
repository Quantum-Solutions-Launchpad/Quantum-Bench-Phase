#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_local.sh"
setup_local_env

OUT_LOG_DIR="${LOG_DIR}/hydrogen/H-linear/ground-state-energy/nqubits-vs-R"
OUT_PLOT_DIR="${PLOT_DIR}/hydrogen/H-linear/ground-state-energy/nqubits-vs-R"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

X_LABEL=n_qubits
X_PARAM=nqubits
X_START="${H_NQUBITS_START:-4}"
X_END="${H_NQUBITS_END:-16}"
X_STEP="${H_NQUBITS_STEP:-4}"
Y_PARAM=R
Y_START="${H_R_START:-0.5}"
Y_END="${H_R_END:-1.0}"
Y_STEP="${H_R_STEP:-0.1}"
SWEEP_TAG="nqubits-vs-R"

H_COUNTS="${H_COUNTS:-2 4 6 8}"
H_RADII="${H_RADII:-0.5 0.6 0.7 0.8 0.9 1.0}"
HAMLIB_GEOMETRY="${HAMLIB_GEOMETRY:-linear}"
HAMLIB_BASIS="${HAMLIB_BASIS:-sto-6g}"
HAMLIB_SELECT="${HAMLIB_SELECT:-ham_JW}"
HAMLIB_TAG="${HAMLIB_SELECT//[^A-Za-z0-9]/-}"
HAMLIB_WORK_DIR="${HAMLIB_WORK_DIR:-${TMPDIR:-/tmp}/${USER}-hydrogen-hamlib-nqubits}"

PIPELINE="analytic"

stage_hydrogen_hamlib "${HAMLIB_WORK_DIR}" "${HAMLIB_GEOMETRY}" "${HAMLIB_BASIS}" \
    ${H_COUNTS} -- ${H_RADII}

echo "==================================================================="
echo "Hydrogen linear ground-state energy n_qubits vs R - Analytic (local)"
echo "  sweep:       ${X_LABEL} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  hamlib root: ${HAMLIB_ROOT:-${HOME}/hamlib/chemistry/electronic/hydrogen_data}"
echo "  staged in:   ${HAMLIB_WORK_DIR} (${#STAGED_HAMLIB_FILES[@]} files)"
echo "  select:      ${HAMLIB_SELECT}"
echo "  methods:     analytic"
echo "  workers:     ${QBP_JOBS_PER_SHARD}"
echo "==================================================================="

cmd=(
    "${QBP_CLI}" run
    --method analytic
    --observable E
    --qubit-operator "${STAGED_HAMLIB_FILES[@]}"
    --select "${HAMLIB_SELECT}"
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    --log-path "${OUT_LOG_DIR}/${PIPELINE}-${HAMLIB_TAG}-E-${SWEEP_TAG}.json"
    --plot-path "${OUT_PLOT_DIR}/${PIPELINE}-${HAMLIB_TAG}-E-${SWEEP_TAG}.pdf"
    --hide-plot
)

echo "Running computation..."
echo "Command: ${cmd[*]}"
echo ""

if "${cmd[@]}"; then
    echo "Completed at $(date)"
else
    EXIT_STATUS=$?
    echo "ERROR: Computation failed with status ${EXIT_STATUS}"
    exit "${EXIT_STATUS}"
fi
