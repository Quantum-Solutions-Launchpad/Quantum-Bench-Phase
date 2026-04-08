#!/bin/bash
#SBATCH -J haldane-real-simulate
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

cd /pscratch/sd/m/mbao202/NNL-P7/slurm/haldane

module load python

# numpy/scipy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 4 config in parallel with 1 node each
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 4 --t2 0.0 --no-debug" &
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 4 --t2 0.05 --no-debug" &
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 6 --t2 0.0 --no-debug" &
srun -N 1 -n 1 -c 256 --exclusive bash -c "source venv/bin/activate && python scripts/haldane-model/real_space_simulated_ideal.py --n-sites 6 --t2 0.05 --no-debug" &

wait

echo "All configurations completed at $(date)"
