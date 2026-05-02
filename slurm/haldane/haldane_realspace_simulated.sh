#!/bin/bash
#SBATCH -J haldane-real-simulate
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/logs/slurm/%x-%j.err

REPO_ROOT="/pscratch/sd/m/mbao202/NNL-P7"
source "${REPO_ROOT}/slurm/common/realspace_simulated.sh"
setup_realspace_env

# 4 configs in parallel, each split into one single-core srun task per shard
run_sharded_config "scripts/real_space_simulated_ideal.py --model haldane --n-sites 6 --t2 0.05 --no-debug" &
run_sharded_config "scripts/real_space_simulated_ideal.py --model haldane --n-sites 4 --t2 0.5 --no-debug" &
run_sharded_config "scripts/real_space_simulated_ideal.py --model haldane --n-sites 6 --t2 1.0 --no-debug" &
run_sharded_config "scripts/real_space_simulated_ideal.py --model haldane --n-sites 6 --t2 0.5 --no-debug" &

wait

echo "All configurations completed at $(date)"
