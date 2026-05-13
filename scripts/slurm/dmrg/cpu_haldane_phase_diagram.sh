#!/bin/bash
#SBATCH -A m5027
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -J cpu_haldane_phase
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/logs/%x/%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/logs/%x/%j.err

ROOT="/pscratch/sd/m/mbao202/NNL-P7"
JULIA_PROJECT_DIR="${ROOT}/scripts/julia-dmrg"
SCRIPT="${JULIA_PROJECT_DIR}/CPU_dmrg_haldane_phase_diagram.jl"

mkdir -p "${ROOT}/logs/${SLURM_JOB_NAME}"
cd "${ROOT}"
module load julia/1.11.7

export JULIA_PROJECT="${JULIA_PROJECT_DIR}"
export JULIA_DEPOT_PATH="/pscratch/sd/m/mbao202/julia_depot"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${JULIA_NUM_THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export DMRG_NSWEEPS="${DMRG_NSWEEPS:-4}"
export DMRG_MAXDIM="${DMRG_MAXDIM:-20,50,100,200}"
export DMRG_CUTOFF="${DMRG_CUTOFF:-1e-9}"
export DMRG_PHI="${DMRG_PHI:-0.7853981633974483}"
export DMRG_CONSERVE_QNS="${DMRG_CONSERVE_QNS:-true}"
export DMRG_N_OCC="${DMRG_N_OCC:-all}"
export DMRG_SEEDS="${DMRG_SEEDS:-401,402}"
export DMRG_SAVE_BATCH="${DMRG_SAVE_BATCH:-2}"

N_START="${N_START:-10}"
N_END="${N_END:-100}"
N_STEP="${N_STEP:-10}"
T1="${T1:-1.0}"
T2_START="${T2_START:-0.0}"
T2_END="${T2_END:-2.0}"
T2_STEP="${T2_STEP:-0.25}"

export DMRG_N_STEP="${DMRG_N_STEP:-$N_STEP}"

BASE_LABEL="${BASE_LABEL:-cpu_haldane_phase_${SLURM_JOB_ID}}"
mkdir -p "${ROOT}/cache/haldane-model/real-space/dmrg/${BASE_LABEL}"

echo "=========================================="
echo "CPU Haldane DMRG Phase Diagram on Perlmutter"
echo "  Job ID:           ${SLURM_JOB_ID}"
echo "  Nodes:            ${SLURM_JOB_NUM_NODES}"
echo "  Tasks:            ${SLURM_NTASKS}"
echo "  Tasks per node:   ${SLURM_NTASKS_PER_NODE}"
echo "  CPUs per task:    ${SLURM_CPUS_PER_TASK}"
echo "  n_sites range:    ${N_START}-${N_END}"
echo "  n_sites step:     ${DMRG_N_STEP}"
echo "  t1:               ${T1}"
echo "  t2 range:         ${T2_START}:${T2_STEP}:${T2_END}"
echo "  nsweeps:          ${DMRG_NSWEEPS}"
echo "  maxdim:           ${DMRG_MAXDIM}"
echo "  seeds:            ${DMRG_SEEDS}"
echo "  save batch:       ${DMRG_SAVE_BATCH}"
echo "=========================================="

echo "Rank placement:"
srun --label bash -c 'echo "Rank ${SLURM_PROCID} on $(hostname -s) (threads=${SLURM_CPUS_PER_TASK})"' | sort -n
echo ""

julia --project="${JULIA_PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

srun --cpu-bind=cores \
  julia --project="${JULIA_PROJECT_DIR}" \
  "${SCRIPT}" "${N_START}" "${N_END}" "${T1}" "${T2_START}" "${T2_END}" "${T2_STEP}" "${BASE_LABEL}"

echo "Done. Results in ${ROOT}/cache/haldane-model/real-space/dmrg/${BASE_LABEL}"
