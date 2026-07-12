#!/bin/bash
#SBATCH -J haldane-2x2-E-nocc-t2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 20:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Unified parallel runner: Haldane 2x2 ground-state energy n_occ vs t2 sweep.
# Runs all methods (iqpe, vqe, dmrg, analytic) in a SINGLE qbp run call
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
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-8}"

export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/manuscript-plots/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/manuscript-plots/plots}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

OUT_LOG_DIR="${LOG_DIR}/haldane/2x2/ground-state-energy/nocc-vs-t2"
OUT_PLOT_DIR="${PLOT_DIR}/haldane/2x2/ground-state-energy/nocc-vs-t2"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

X_PARAM=n_occ
X_START="${HALDANE_N_OCC_START:-0}"
X_END="${HALDANE_N_OCC_END:-8}"
X_STEP="${HALDANE_N_OCC_STEP:-1}"
Y_PARAM=t2
Y_START="${HALDANE_T2_START:-0.0}"
Y_END="${HALDANE_T2_END:-1.0}"
Y_STEP="${HALDANE_T2_STEP:-0.1}"
SWEEP_TAG="nocc-vs-t2"

PIPELINE="${PIPELINE:-ideal}"
backend_args=()
if [[ "${PIPELINE}" == "noisy" ]]; then
    backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
fi

fixed_args=(
    --t1 "${HALDANE_T1:-1.0}"
    --phi "${HALDANE_PHI:-0.7853981633974483}"
    --M "${HALDANE_M:-0.0}"
)

echo "==================================================================="
echo "Haldane 2x2 ground-state energy n_occ vs t2 (SHARDED)"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  methods:     iqpe vqe dmrg analytic"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "==================================================================="

cmd=(
    --model haldane
    --method iqpe vqe dmrg analytic
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
append_dmrg_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-${PIPELINE}-all-E-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-all-E-${SWEEP_TAG}.pdf"

run_visualizer_sharded_cmd cmd
EXIT_STATUS=$?
