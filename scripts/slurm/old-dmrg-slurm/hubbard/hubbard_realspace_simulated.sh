#!/bin/bash
#SBATCH -J hubbard-real-simulate
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/logs/slurm/%x-%j.err

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/old-dmrg-slurm/common/realspace_simulated.sh"
setup_realspace_env

run_sharded_config "scripts/real_space_simulated_ideal.py --model hubbard --n-sites 4 --U 2.0 --no-debug" &
run_sharded_config "scripts/real_space_simulated_ideal.py --model hubbard --n-sites 4 --U 1.0 --no-debug" &
run_sharded_config "scripts/real_space_simulated_ideal.py --model hubbard --n-sites 6 --U 2.0 --no-debug" &
run_sharded_config "scripts/real_space_simulated_ideal.py --model hubbard --n-sites 6 --U 1.0 --no-debug" &
wait

echo "All configurations completed at $(date)"
