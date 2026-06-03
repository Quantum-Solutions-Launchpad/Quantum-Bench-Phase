#!/bin/bash
#SBATCH -J haldane-2x2-dmrg
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}-mpl}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/pscratch/sd/m/mbao202/julia_depot}"
DMRG_THREADS="${DMRG_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-${DMRG_THREADS}}"
export OMP_NUM_THREADS="${DMRG_OMP_NUM_THREADS:-${DMRG_THREADS}}"
export MKL_NUM_THREADS="${DMRG_MKL_NUM_THREADS:-${DMRG_THREADS}}"
export OPENBLAS_NUM_THREADS="${DMRG_OPENBLAS_NUM_THREADS:-${DMRG_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/examples/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/examples/plots}"
PHI="${PHI:-0.7853981633974483}"

mkdir -p "${LOG_DIR}" "${PLOT_DIR}" "${REPO_ROOT}/scripts/logs/slurm"

echo "Running Haldane 2x2 ITensorMPS DMRG phase diagram against exact values"
echo "  job id:      ${SLURM_JOB_ID:-manual}"
echo "  nodes:       ${SLURM_JOB_NUM_NODES:-local}"
echo "  shards:      ${SHARDS}"
echo "  log dir:     ${LOG_DIR}"
echo "  plot dir:    ${PLOT_DIR}"
echo "  t2 sweep:    ${HALDANE_T2_START:-0.0} ${HALDANE_T2_END:-1.0} ${HALDANE_T2_STEP:-0.1}"
echo "  nsweeps:     ${DMRG_NSWEEPS:-4}"
echo "  maxdims:     ${DMRG_MAXDIMS:-20,50,100,200}"
echo "  cutoff:      ${DMRG_CUTOFF:-1e-9}"
echo "  threads:     ${DMRG_THREADS} per shard"

cmd=(
    dmrg
    --model haldane
    --lattice 2 2
    --x-param n_occ
    --y-param t2
    --y-range "${HALDANE_T2_START:-0.0}" "${HALDANE_T2_END:-1.0}" "${HALDANE_T2_STEP:-0.1}"
    --t1 "${HALDANE_T1:-1.0}"
    --phi "${HALDANE_PHI:-${PHI}}"
    --M "${HALDANE_M:-0.0}"
    --log-dir "${LOG_DIR}"
    --plot-dir "${PLOT_DIR}"
    --hide-plot
    --nsweeps "${DMRG_NSWEEPS:-4}"
    --maxdims "${DMRG_MAXDIMS:-20,50,100,200}"
    --cutoff "${DMRG_CUTOFF:-1e-9}"
    --seed "${DMRG_SEED:-1234}"
)

printf -v cmd_string "%q " "${cmd[@]}"
run_sharded_config "${cmd_string}"

echo "Completed Haldane 2x2 DMRG run at $(date)"
echo "  summary: ${LOG_DIR}/haldane/2x2/dmrg/dmrg-n_occ-vs-t2.json"
echo "  raw dir: ${LOG_DIR}/haldane/2x2/dmrg/raw-data"
echo "  plot:    ${PLOT_DIR}/haldane/2x2/dmrg/dmrg-n_occ-vs-t2.pdf"
