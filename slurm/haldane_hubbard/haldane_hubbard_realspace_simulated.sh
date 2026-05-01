#!/bin/bash
#SBATCH -J haldane-hubbard-real-simulate
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/logs/slurm/%x-%j.err

REPO_ROOT="/pscratch/sd/m/mbao202/NNL-P7"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"
# prevents each srun task from spawning lots of internal threads and oversubscribing CPUs
export OMP_NUM_THREADS=1 # 1 thread per process for NumPy/SciPy-enabled code
export MKL_NUM_THREADS=1 # BLAS/LAPACK backend uses 1 thread per process
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Avoid joblib loky multiprocessing issues on this Python stack
export JOBLIB_MULTIPROCESSING=0
export MPLCONFIGDIR="/tmp/${USER}-mpl"

# 4 config in parallel with 1 node each
srun -N 1 -n 1 -c 128 --exclusive bash -c "python scripts/real_space_simulated_ideal.py --model haldane-hubbard --n-sites 4 --U 1.0 --t2 0.0 --no-debug" &
srun -N 1 -n 1 -c 128 --exclusive bash -c "python scripts/real_space_simulated_ideal.py --model haldane-hubbard --n-sites 4 --U 1.0 --t2 0.05 --no-debug" &
srun -N 1 -n 1 -c 128 --exclusive bash -c "python scripts/real_space_simulated_ideal.py --model haldane-hubbard --n-sites 6 --U 1.0 --t2 0.0 --no-debug" &
srun -N 1 -n 1 -c 128 --exclusive bash -c "python scripts/real_space_simulated_ideal.py --model haldane-hubbard --n-sites 6 --U 1.0 --t2 0.05 --no-debug" &

wait

echo "All configurations completed at $(date)"
