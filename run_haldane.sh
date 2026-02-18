#!/bin/bash
#SBATCH -J haldane-real-space
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 48:00:00
#SBATCH -A m4673
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Navigate to project directory
cd /pscratch/sd/m/mbao202/NNL-P7

# Load modules
module load python

# Prevent numpy/scipy from competing with joblib for cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run 4 configurations in parallel (1 node each)
# Each srun activates venv in a subshell
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 4 --t2 0.0 --no-debug" &
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 4 --t2 0.05 --no-debug" &
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 6 --t2 0.0 --no-debug" &
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 6 --t2 0.05 --no-debug" &

# Wait for all background jobs to complete
wait

echo "All configurations completed at $(date)"
