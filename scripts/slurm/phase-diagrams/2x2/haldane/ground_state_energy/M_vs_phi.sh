#!/bin/bash
#SBATCH -J haldane-2x2-E-M-phi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH -t 20:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Haldane 2x2 ground-state energy: M vs phi sweep.
# PIPELINE=noisy sbatch ... for the noisy pipeline (default: ideal).

set -euo pipefail

export OUT_SUBDIR=ground-state-energy/M-vs-phi
export X_PARAM=M
export X_START="${HALDANE_M_START:--6.0}"
export X_END="${HALDANE_M_END:-6.0}"
export X_STEP="${HALDANE_M_STEP:-0.5}"
export Y_PARAM=phi
export Y_START="${HALDANE_PHI_START:--3.141592653589793}"
export Y_END="${HALDANE_PHI_END:-3.141592653589793}"
export Y_STEP="${HALDANE_PHI_STEP:-0.39269908169872414}"

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
exec bash "${REPO_ROOT}/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/common.sh"
