#!/bin/bash
#SBATCH -J h-linear-vqe-iqpe
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Hydrogen linear systems: n=2,4,6,8,10,12 and R=0.5,0.8,1.1,1.4,1.7,2.0
# Python-based runner with full control over parameters

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

export LOKY_DISABLE_RESOURCE_TRACKER=1
export MPLBACKEND=Agg

# VQE/IQPE parameters (same as haldane)
export VQE_ITERS="${VQE_ITERS:-10000}"
export VQE_LAYERS="${VQE_LAYERS:-5}"
export VQE_REPS="${VQE_REPS:-10}"

export IQPE_TIME="${IQPE_TIME:-0.2}"
export IQPE_TROT="${IQPE_TROT:-5}"
export IQPE_ITERS="${IQPE_ITERS:-8}"
export IQPE_REPS="${IQPE_REPS:-20}"
export IQPE_INITIAL_VQE_N_LAYERS="${IQPE_INITIAL_VQE_N_LAYERS:-2}"
export IQPE_INITIAL_VQE_REPS="${IQPE_INITIAL_VQE_REPS:-2}"
export IQPE_INITIAL_VQE_MAX_ITERS="${IQPE_INITIAL_VQE_MAX_ITERS:-1000}"

# Hamlib path (remote NERSC chemistry dataset)
export HAMLIB_PATH="${HAMLIB_PATH:-https://portal.nersc.gov/cfs/m888/dcamps/hamlib_v1.0/chemistry/hydrogen_linear/H_linear.zip}"

echo "==========================================================================="
echo "Hydrogen Linear: Ground-State Energy VQE + IQPE Sweep"
echo "  n: 2, 4, 6, 8, 10, 12"
echo "  R: 0.5, 0.8, 1.1, 1.4, 1.7, 2.0"
echo "  Methods: VQE (iters=${VQE_ITERS}, layers=${VQE_LAYERS}, reps=${VQE_REPS})"
echo "           IQPE (time=${IQPE_TIME}, trot=${IQPE_TROT}, iters=${IQPE_ITERS}, reps=${IQPE_REPS})"
echo "  Backend: Simulated Ideal"
echo "  Hamlib:  ${HAMLIB_PATH}"
echo "==========================================================================="

python3 "${REPO_ROOT}/scripts/run_hydrogen_vqe_iqpe.py"

echo "Completed at $(date)"
