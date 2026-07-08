#!/bin/bash
#SBATCH -J hubbard-2x2-mag-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH --array=0-1
#SBATCH -t 20:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.err

# Hubbard 2x2 magnetization: n_occ vs U sweep for M_stag/M_total.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"

export OUT_SUBDIR=magnetization/dmrg
export METHODS=dmrg
export HEATMAP="${HEATMAP:-1}"

exec bash "${REPO_ROOT}/scripts/slurm/phase-diagrams/2x2/hubbard/magnetization/common.sh"
