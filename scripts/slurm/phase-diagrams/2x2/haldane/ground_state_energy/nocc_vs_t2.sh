#!/bin/bash
#SBATCH -J haldane-2x2-E-nocc-t2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH -t 20:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Haldane 2x2 ground-state energy: n_occ vs t2 sweep (fixed t1, M, phi=pi/4).
# PIPELINE=noisy sbatch ... for the noisy pipeline (default: ideal).

set -euo pipefail

export OUT_SUBDIR=ground-state-energy/nocc-vs-t2
export X_PARAM=n_occ
export X_START="${HALDANE_N_OCC_START:-0}"
export X_END="${HALDANE_N_OCC_END:-8}"
export X_STEP="${HALDANE_N_OCC_STEP:-1}"
export Y_PARAM=t2
export Y_START="${HALDANE_T2_START:-0.0}"
export Y_END="${HALDANE_T2_END:-1.0}"
export Y_STEP="${HALDANE_T2_STEP:-0.1}"
export SWEEP_TAG=nocc-vs-t2  # keep legacy filename tag so old progress logs resume

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
exec bash "${REPO_ROOT}/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/common.sh"
