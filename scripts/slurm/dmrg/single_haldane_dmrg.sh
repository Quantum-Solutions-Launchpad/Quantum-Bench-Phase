#!/bin/bash
#SBATCH -A m5027
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -J single_haldane_dmrg
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/%x/%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/%x/%j.err


PROJECT_DIR="${PROJECT_DIR:-/pscratch/sd/m/mbao202/NNL-P7}"
JULIA_PROJECT_DIR="${JULIA_PROJECT_DIR:-$PROJECT_DIR/scripts/julia-dmrg}"
SCRIPT="${SCRIPT:-$PROJECT_DIR/scripts/julia-dmrg/dmrg_haldane.jl}"

mkdir -p "${PROJECT_DIR}/logs/${SLURM_JOB_NAME}"
cd "${PROJECT_DIR}"
module load julia/1.11.7

export JULIA_PROJECT="${JULIA_PROJECT_DIR}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/pscratch/sd/m/mbao202/julia_depot}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache-${SLURM_JOB_ID}}"

N_SITES="${N_SITES:-8}"
T1_VALUE="${T1_VALUE:-1.0}"
T2_VALUE="${T2_VALUE:-1.0}"
PHI_VALUE="${PHI_VALUE:-0.7853981633974483}"
MAXDIMS="${MAXDIMS:-20,50,100,200}"
N_OCC_SPEC="${N_OCC_SPEC:-all}"
NSWEEPS="${NSWEEPS:-4}"
CUTOFF="${CUTOFF:-1e-9}"
CONSERVE_QNS="${CONSERVE_QNS:-true}"
SEEDS="${SEEDS:-401,402}"
OUTPUT_JSON="${OUTPUT_JSON:-cache/haldane-model/real-space/dmrg/single-runs/n-sites-${N_SITES}_t2-${T2_VALUE}_t1-${T1_VALUE}.json}"

mkdir -p "$MPLCONFIGDIR"
mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "=========================================="
echo "Single Haldane DMRG Run on Perlmutter"
echo "  Job ID:         ${SLURM_JOB_ID}"
echo "  Node count:     ${SLURM_JOB_NUM_NODES}"
echo "  Tasks:          ${SLURM_NTASKS}"
echo "  CPUs per task:  ${SLURM_CPUS_PER_TASK}"
echo "  n_sites:        ${N_SITES}"
echo "  t1:             ${T1_VALUE}"
echo "  t2:             ${T2_VALUE}"
echo "  phi:            ${PHI_VALUE}"
echo "  maxdims:        ${MAXDIMS}"
echo "  n_occ:          ${N_OCC_SPEC}"
echo "  nsweeps:        ${NSWEEPS}"
echo "  cutoff:         ${CUTOFF}"
echo "  conserve_qns:   ${CONSERVE_QNS}"
echo "  seeds:          ${SEEDS}"
echo "  output:         ${OUTPUT_JSON}"
echo "=========================================="

julia --project="${JULIA_PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

srun --cpu-bind=cores \
  julia --project="${JULIA_PROJECT_DIR}" \
  "${SCRIPT}" \
  --n-sites "${N_SITES}" \
  --t1 "${T1_VALUE}" \
  --t2 "${T2_VALUE}" \
  --phi "${PHI_VALUE}" \
  --maxdims "${MAXDIMS}" \
  --n-occ "${N_OCC_SPEC}" \
  --nsweeps "${NSWEEPS}" \
  --cutoff "${CUTOFF}" \
  --conserve-qns "${CONSERVE_QNS}" \
  --seeds "${SEEDS}" \
  --output "${OUTPUT_JSON}"

echo "Done. Results in ${PROJECT_DIR}/${OUTPUT_JSON}"
