#!/bin/bash
#SBATCH -J h2-linear-vqe-iqpe
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 20:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Hydrogen linear systems: n=2,4,6,8,10,12 and R=0.5,0.8,1.1,1.4,1.7,2.0
# Runs VQE and IQPE methods with same parameters as haldane runs

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
SHARDS="${SHARDS:-${SLURM_NTASKS:-16}}"
setup_visualizer_env

export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-1}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs/new-data}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots/new-data}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

OUT_LOG_DIR="${LOG_DIR}/hydrogen/H-linear/ground-state-energy/n-vs-R"
OUT_PLOT_DIR="${PLOT_DIR}/hydrogen/H-linear/ground-state-energy/n-vs-R"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

X_PARAM=n
X_START="${H_N_START:-2}"
X_END="${H_N_END:-12}"
X_STEP="${H_N_STEP:-2}"
Y_PARAM=R
Y_START="${H_R_START:-0.5}"
Y_END="${H_R_END:-2.0}"
Y_STEP="${H_R_STEP:-0.3}"
SWEEP_TAG="n-vs-R"

# Hamlib chemistry hydrogen linear models
HAMLIB_BASE="${HAMLIB_BASE:-https://portal.nersc.gov/cfs/m888/dcamps/hamlib_v1.0/chemistry}"
HAMLIB_PATH="${HAMLIB_PATH:-${HAMLIB_BASE}/hydrogen_linear/H_linear.zip}"

PIPELINE="simulated-ideal"

echo "==================================================================="
echo "Hydrogen linear ground-state energy n vs R (SHARDED) - VQE + IQPE"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  hamlib:      ${HAMLIB_PATH}"
echo "  methods:     vqe iqpe"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "  output:      new-data"
echo "==================================================================="

cmd=(
    --method vqe iqpe
    --qubit-operator "${HAMLIB_PATH}"
    --observable E
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    --select H-linear
)

append_vqe_args cmd
append_iqpe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/${PIPELINE}-vqe-iqpe-E-n-vs-R.json" "${OUT_PLOT_DIR}/${PIPELINE}-vqe-iqpe-E-n-vs-R.pdf"

if run_visualizer_sharded_cmd cmd; then
    echo "Completed at $(date)"
else
    EXIT_STATUS=$?
    echo "ERROR: Computation failed with status ${EXIT_STATUS}"
    exit "${EXIT_STATUS}"
fi
