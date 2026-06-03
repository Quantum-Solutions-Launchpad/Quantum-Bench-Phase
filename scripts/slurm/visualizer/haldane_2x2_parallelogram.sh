#!/bin/bash
#SBATCH -J haldane-2x2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/common/realspace_simulated.sh"
setup_realspace_env

export MPLBACKEND="${MPLBACKEND:-Agg}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/examples/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/examples/plots}"
PHI="${PHI:-0.7853981633974483}"

mkdir -p "${LOG_DIR}" "${PLOT_DIR}" "${REPO_ROOT}/scripts/logs/slurm"

cmd=(
    simulated-ideal
    --model haldane
    --lattice 2 2
    --x-param n_occ
    --y-param t2
    --y-range "${HALDANE_T2_START:-0.0}" "${HALDANE_T2_END:-1.0}" "${HALDANE_T2_STEP:-0.1}"
    --t1 "${HALDANE_T1:-1.0}"
    --phi "${HALDANE_PHI:-${PHI}}"
    --M "${HALDANE_M:-0.0}"
    --vqe-iters "${VQE_ITERS:-10000}"
    --vqe-layers "${VQE_LAYERS:-5}"
    --vqe-reps "${VQE_REPS:-10}"
    --iqpe-time "${IQPE_TIME:-0.2}"
    --iqpe-trot "${IQPE_TROT:-5}"
    --iqpe-iters "${IQPE_ITERS:-8}"
    --iqpe-reps "${IQPE_REPS:-20}"
    --log-dir "${LOG_DIR}"
    --plot-dir "${PLOT_DIR}"
    --hide-plot
    --hide-legend
)

echo "Running Haldane 2x2 parallelogram phase diagram"
echo "  job id:      ${SLURM_JOB_ID:-manual}"
echo "  nodes:       ${SLURM_JOB_NUM_NODES:-local}"
echo "  shards:      ${SHARDS}"
echo "  t2 sweep:    ${HALDANE_T2_START:-0.0} ${HALDANE_T2_END:-1.0} ${HALDANE_T2_STEP:-0.1}"
echo "  log dir:     ${LOG_DIR}"
echo "  plot dir:    ${PLOT_DIR}"

printf -v cmd_string "%q " "${cmd[@]}"
run_sharded_config "${cmd_string}"

echo "Completed Haldane 2x2 run at $(date)"
echo "  summary: ${LOG_DIR}/haldane/2x2/simulated-ideal-3d-n_occ-vs-t2.json"
echo "  raw:     ${LOG_DIR}/haldane/2x2/raw-data/simulated-ideal-3d-n_occ-vs-t2.json"
echo "  plot:    ${PLOT_DIR}/haldane/2x2/simulated-ideal-3d-n_occ-vs-t2.pdf"
