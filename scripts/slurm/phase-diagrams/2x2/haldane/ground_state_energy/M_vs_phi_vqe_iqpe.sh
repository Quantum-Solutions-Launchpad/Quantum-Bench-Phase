#!/bin/bash
#SBATCH -J haldane-2x2-E-M-phi-vqe-iqpe
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Unified parallel runner: Haldane 2x2 ground-state energy M vs phi sweep.
# Runs VQE and IQPE methods only in a SINGLE qbp run call
# with joblib.Parallel() scheduling ALL jobs simultaneously.
# PIPELINE=noisy sbatch ... for the noisy pipeline (default: ideal).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
SHARDS="${SHARDS:-${SLURM_NTASKS:-16}}"
setup_visualizer_env

export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs/new-data}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots/new-data}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

OUT_LOG_DIR="${LOG_DIR}/haldane/2x2/ground-state-energy/M-vs-phi"
OUT_PLOT_DIR="${PLOT_DIR}/haldane/2x2/ground-state-energy/M-vs-phi"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

X_PARAM=M
X_START="${HALDANE_M_START:--6.0}"
X_END="${HALDANE_M_END:-6.0}"
X_STEP="${HALDANE_M_STEP:-0.5}"
Y_PARAM=phi
Y_START="${HALDANE_PHI_START:--3.141592653589793}"
Y_END="${HALDANE_PHI_END:-3.141592653589793}"
Y_STEP="${HALDANE_PHI_STEP:-0.39269908169872414}"
SWEEP_TAG="M-vs-phi"

PIPELINE="${PIPELINE:-ideal}"
backend_args=()
if [[ "${PIPELINE}" == "noisy" ]]; then
    backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
fi

fixed_args=(
    --t1 "${HALDANE_T1:-1.0}"
    --t2 "${HALDANE_T2:-1.0}"
)

echo "==================================================================="
echo "Haldane 2x2 ground-state energy M vs phi (SHARDED) - VQE + IQPE"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM}"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  methods:     iqpe vqe"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "  output:      new-data"
echo "==================================================================="

cmd=(
    --model haldane
    --method iqpe vqe
    --lattice 2 2
    --observable E
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    "${fixed_args[@]}"
)

if [[ "${PIPELINE}" == "noisy" ]]; then
    cmd+=("${backend_args[@]}")
fi

append_vqe_args cmd
append_iqpe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-${PIPELINE}-vqe-iqpe-E-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-vqe-iqpe-E-${SWEEP_TAG}.pdf"

run_visualizer_sharded_cmd cmd
EXIT_STATUS=$?
