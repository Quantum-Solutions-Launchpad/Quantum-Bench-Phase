#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs/new-data}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots/new-data}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

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

HAMLIB_FILES="${HAMLIB_FILES:-/pscratch/sd/m/mbao202/ES_H2_linear_R0.5_sto-6g.hdf5 /pscratch/sd/m/mbao202/ES_H2_linear_R0.6_sto-6g.hdf5 /pscratch/sd/m/mbao202/ES_H2_linear_R0.7_sto-6g.hdf5 /pscratch/sd/m/mbao202/ES_H2_linear_R0.8_sto-6g.hdf5 /pscratch/sd/m/mbao202/ES_H2_linear_R0.9_sto-6g.hdf5 /pscratch/sd/m/mbao202/ES_H2_linear_R1.0_sto-6g.hdf5 /pscratch/sd/m/mbao202/ES_H4_linear_R0.5_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H4_linear_R0.6_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H4_linear_R0.7_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H4_linear_R0.8_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H4_linear_R0.9_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H4_linear_R1.0_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H6_linear_R0.5_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H6_linear_R0.6_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H6_linear_R0.7_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H6_linear_R0.8_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H6_linear_R0.9_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H6_linear_R1.0_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H8_linear_R0.5_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H8_linear_R0.6_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H8_linear_R0.7_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H8_linear_R0.8_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H8_linear_R0.9_sto-6g_ham.hdf5 /pscratch/sd/m/mbao202/ES_H8_linear_R1.0_sto-6g_ham.hdf5}"
HAMLIB_SELECT="${HAMLIB_SELECT:-ham_JW}"

PIPELINE="dmrg"

echo "==================================================================="
echo "Hydrogen linear ground-state energy n_qubits vs R - DMRG (CPU Interactive)"
echo "  sweep:       ${X_LABEL} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  hamlib:      ${HAMLIB_FILES}"
echo "  select:      ${HAMLIB_SELECT}"
echo "  method:      dmrg"
echo "  compute:     CPU node (interactive)"
echo "==================================================================="

cmd=(
    qbp run
    --method dmrg
    --observable E
)

HAMLIB_WORK_DIR="${HAMLIB_WORK_DIR:-/tmp/${USER}-hydrogen-hamlib-nqubits}"
mkdir -p "${HAMLIB_WORK_DIR}"

# qbp extracts sweep variables from file-name tokens. The source files encode
# molecule size as H2/H4/...; expose the qubit count as nqubits4/nqubits8/...
# via symlinks while leaving the original HamLib files untouched.
declare -a QBP_HAMLIB_FILES=()
for hamlib in ${HAMLIB_FILES}; do
    base="$(basename "${hamlib}")"
    if [[ ! "${base}" =~ ES_H([0-9]+)_ ]]; then
        echo "ERROR: Cannot infer H count from HamLib filename: ${hamlib}" >&2
        exit 2
    fi
    h_count="${BASH_REMATCH[1]}"
    nqubits=$((2 * h_count))
    link_base="${base/ES_H${h_count}_/ES_nqubits${nqubits}_}"
    link_path="${HAMLIB_WORK_DIR}/${link_base}"
    ln -sfn "${hamlib}" "${link_path}"
    QBP_HAMLIB_FILES+=("${link_path}")
done

cmd+=(--qubit-operator)
for hamlib in "${QBP_HAMLIB_FILES[@]}"; do
    cmd+=("${hamlib}")
done

cmd+=(
    --select "${HAMLIB_SELECT}"
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    --dmrg-nsweeps "${DMRG_NSWEEPS:-4}"
    --dmrg-maxdims "${DMRG_MAXDIMS:-20,50,100,200}"
    --dmrg-cutoff "${DMRG_CUTOFF:-1e-9}"
    --dmrg-seed "${DMRG_SEED:-1234}"
    --log-path "${OUT_LOG_DIR}/${PIPELINE}-E-${SWEEP_TAG}.json"
    --plot-path "${OUT_PLOT_DIR}/${PIPELINE}-E-${SWEEP_TAG}.pdf"
    --hide-plot
)

echo "Running: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo "Completed at $(date)"
    echo ""
    echo "==================================================================="
    echo "Output files:"
    echo "  JSON: ${OUT_LOG_DIR}/${PIPELINE}-E-${SWEEP_TAG}.json"
    echo "  PDF:  ${OUT_PLOT_DIR}/${PIPELINE}-E-${SWEEP_TAG}.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo "ERROR: Computation failed with status ${EXIT_STATUS}"
    exit "${EXIT_STATUS}"
fi
