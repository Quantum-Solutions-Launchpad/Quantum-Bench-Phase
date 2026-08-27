#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_local.sh"
setup_local_env
setup_local_dmrg_env

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

DMRG_NSWEEPS="${DMRG_NSWEEPS:-24}"
DMRG_MAXDIMS="${DMRG_MAXDIMS:-32,64,128,$(printf '256,%.0s' $(seq 1 20))256}"
DMRG_CUTOFF="${DMRG_CUTOFF:-1e-13}"
DMRG_SEED="${DMRG_SEED:-1234}"
DMRG_INITIAL_STATE="${DMRG_INITIAL_STATE:-random}"
DMRG_INITIAL_LINKDIM="${DMRG_INITIAL_LINKDIM:-32}"
DMRG_NOISE="${DMRG_NOISE:-1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,$(printf '0,%.0s' $(seq 1 17))0}"
DMRG_MPO_CUTOFF="${DMRG_MPO_CUTOFF:-1e-18}"
DMRG_KRYLOVDIM="${DMRG_KRYLOVDIM:-0}"

PIPELINE="dmrg"

stage_hydrogen_hamlib "${HAMLIB_WORK_DIR}" "${HAMLIB_GEOMETRY}" "${HAMLIB_BASIS}" \
    ${H_COUNTS} -- ${H_RADII}

echo "==================================================================="
echo "Hydrogen linear ground-state energy n_qubits vs R - DMRG (local)"
echo "  sweep:       ${X_LABEL} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  hamlib root: ${HAMLIB_ROOT:-${HOME}/hamlib/chemistry/electronic/hydrogen_data}"
echo "  staged in:   ${HAMLIB_WORK_DIR} (${#STAGED_HAMLIB_FILES[@]} files)"
echo "  select:      ${HAMLIB_SELECT}"
echo "  method:      dmrg (nsweeps=${DMRG_NSWEEPS}, init=${DMRG_INITIAL_STATE})"
echo "  workers:     ${QBP_JOBS_PER_SHARD}"
echo "==================================================================="

cmd=(
    "${QBP_CLI}" run
    --method dmrg
    --observable E
    --qubit-operator "${STAGED_HAMLIB_FILES[@]}"
    --select "${HAMLIB_SELECT}"
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    --dmrg-nsweeps "${DMRG_NSWEEPS}"
    --dmrg-maxdims "${DMRG_MAXDIMS}"
    --dmrg-cutoff "${DMRG_CUTOFF}"
    --dmrg-seed "${DMRG_SEED}"
    --dmrg-initial-state "${DMRG_INITIAL_STATE}"
    --dmrg-initial-linkdim "${DMRG_INITIAL_LINKDIM}"
    --dmrg-noise "${DMRG_NOISE}"
    --dmrg-mpo-cutoff "${DMRG_MPO_CUTOFF}"
    --dmrg-eigsolve-krylovdim "${DMRG_KRYLOVDIM}"
    --dmrg-julia "${JULIA_BIN}"
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
