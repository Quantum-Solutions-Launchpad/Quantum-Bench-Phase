#!/bin/bash
#SBATCH -J haldane-2x2-band-noisy-vqe
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Haldane 2x2 band structure kx vs ky sweep (noisy pipeline) - VQE only
# Runs VQE method separately for band structure visualization

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

OUT_LOG_DIR="${LOG_DIR}/haldane/2x2/band-structure-noisy"
OUT_PLOT_DIR="${PLOT_DIR}/haldane/2x2/band-structure-noisy"
mkdir -p "${OUT_LOG_DIR}" "${OUT_PLOT_DIR}"

X_PARAM=kx
X_START="${HALDANE_KX_START:--3.141592653589793}"
X_END="${HALDANE_KX_END:-3.141592653589793}"
X_STEP="${HALDANE_KX_STEP:-0.39269908169872414}"
Y_PARAM=ky
Y_START="${HALDANE_KY_START:--3.141592653589793}"
Y_END="${HALDANE_KY_END:-3.141592653589793}"
Y_STEP="${HALDANE_KY_STEP:-0.39269908169872414}"
SWEEP_TAG="kx-vs-ky"

PIPELINE="${PIPELINE:-noisy}"
backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")

fixed_args=(
    --t1 "${HALDANE_T1:-1.0}"
    --t2 "${HALDANE_T2:-0.1}"
    --phi "${HALDANE_PHI:-0.7853981633974483}"
    --M "${HALDANE_M:-0.0}"
)

echo "==================================================================="
echo "Haldane 2x2 band structure kx vs ky (noisy) - VQE"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  backend:     ${NOISY_BACKEND:-FakeSherbrooke}"
echo "  methods:     vqe"
echo "  grid points: ~289 (17 × 17)"
echo "  parallelism: distributed across ${SHARDS} shards on multiple nodes"
echo "  output:      new-data"
echo "==================================================================="

cmd=(
    --model haldane-honeycomb
    --method vqe
    --observable E
    --heatmap
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    "${fixed_args[@]}"
    "${backend_args[@]}"
    --vqe-layers "${VQE_LAYERS:-6}"
    --vqe-reps "${VQE_REPS:-5}"
    --vqe-iters "${VQE_ITERS:-400}"
)
append_qbp_output_paths cmd "${OUT_LOG_DIR}/simulated-${PIPELINE}-vqe-E-${SWEEP_TAG}.json" "${OUT_PLOT_DIR}/simulated-${PIPELINE}-vqe-E-${SWEEP_TAG}.pdf"

if run_visualizer_sharded_cmd cmd; then
    echo "Completed at $(date)"
else
    EXIT_STATUS=$?
    echo "ERROR: Computation failed with status ${EXIT_STATUS}"
    exit "${EXIT_STATUS}"
fi
