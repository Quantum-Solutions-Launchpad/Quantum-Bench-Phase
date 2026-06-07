#!/bin/bash
#SBATCH -A m5027
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -J haldane_n1000_t2_0
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/logs/%x/%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/logs/%x/%j.err

PROJECT_DIR="/pscratch/sd/m/mbao202/NNL-P7"
JULIA_PROJECT_DIR="${PROJECT_DIR}/scripts/julia-dmrg"
SCRIPT="${PROJECT_DIR}/scripts/julia-dmrg/old-haldane-only/dmrg_haldane.jl"

mkdir -p "${PROJECT_DIR}/logs/${SLURM_JOB_NAME}"
cd "${PROJECT_DIR}"
module load julia/1.11.7

export JULIA_PROJECT="${JULIA_PROJECT_DIR}"
export JULIA_DEPOT_PATH="/pscratch/sd/m/mbao202/julia_depot"
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl-cache-${SLURM_JOB_ID}"

N_SITES=120
T1_VALUE=1.0
T2_VALUE=0.0
PHI_VALUE=0.7853981633974483
MAXDIMS=20,50,100,200
N_OCC_SPEC=all
NSWEEPS=4
CUTOFF=1e-9
CONSERVE_QNS=true
SEEDS=401,402
OUTPUT_JSON="cache/haldane-model/real-space/dmrg/large-system-runs/n-sites-1000_t2-0_t1-1.0.json"

mkdir -p "$MPLCONFIGDIR"
mkdir -p "$(dirname "$OUTPUT_JSON")"

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
