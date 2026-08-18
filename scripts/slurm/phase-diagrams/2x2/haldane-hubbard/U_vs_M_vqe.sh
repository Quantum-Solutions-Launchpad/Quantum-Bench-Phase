#!/bin/bash
#SBATCH -J hh-2x2-E-M-U-vqe
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH -t 20:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Unified parallel runner: Haldane-Hubbard 2x2 ground-state energy M vs U sweep.
# Runs VQE method in a SINGLE qbp run call
# with joblib.Parallel() scheduling ALL jobs simultaneously.
# Always runs with a noisy backend; set NOISY_BACKEND to override FakeSherbrooke.

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

OUT_LOG_DIR="${LOG_DIR}/haldane/2x2/ground-state-energy/M-vs-U"
OUT_PLOT_DIR="${PLOT_DIR}/haldane/2x2/ground-state-energy/M-vs-U"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

X_PARAM=M
X_START="${HALDANE_M_START:-0}"
X_END="${HALDANE_M_END:-3}"
X_STEP="${HALDANE_M_STEP:-0.25}"
Y_PARAM=U
Y_START="${HALDANE_U_START:-0.0}"
Y_END="${HALDANE_U_END:-3.0}"
Y_STEP="${HALDANE_U_STEP:-0.25}"
SWEEP_TAG="M-vs-U"

PIPELINE="noisy"
backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")

fixed_args=(
    --t1 "${HALDANE_T1:-1.0}"
    --t2 "${HALDANE_T2:-1.0}"
    --phi "${HALDANE_PHI:-0.7853981633974483}"
    --n-occ "${HALDANE_N_OCC:-4}"
)

echo "==================================================================="
echo "Haldane-Hubbard 2x2 ground-state energy M vs U (SHARDED) - VQE"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  methods:     vqe"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "  output:      new-data"
echo "==================================================================="

cmd=(
    --model haldane-hubbard
    --method vqe
    --lattice 2 2
    --observable E
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    "${fixed_args[@]}"
)

cmd+=("${backend_args[@]}")

append_vqe_args cmd
append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-${PIPELINE}-vqe-E-M-vs-U.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-vqe-E-M-vs-U.pdf"

if run_visualizer_sharded_cmd cmd; then
    echo "Completed at $(date)"
else
    EXIT_STATUS=$?
    echo "ERROR: Computation failed with status ${EXIT_STATUS}"
    exit "${EXIT_STATUS}"
fi
