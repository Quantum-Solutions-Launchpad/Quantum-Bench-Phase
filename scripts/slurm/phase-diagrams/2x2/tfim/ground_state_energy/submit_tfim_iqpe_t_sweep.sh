#!/bin/bash
#SBATCH -J tfim-1d-E-h-iqpe-Lx8-t-sweep
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A m5027
#SBATCH --array=0-10
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-t%a-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-t%a-%j.err

# Submit IQPE jobs for multiple t values (0.0 to 1.0 step 0.1)
# Each job sweeps h from 0.0 to 6.0 step 0.5 with Lx=8

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
SCRIPT="${REPO_ROOT}/scripts/slurm/phase-diagrams/2x2/tfim/ground_state_energy/tfim_iqpe_t_sweep_h_scan.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Running TFIM IQPE t-sweep array task ${SLURM_ARRAY_TASK_ID:-0}/10"
    exec bash "${SCRIPT}"
fi

echo "Submitting TFIM IQPE t-sweep as a Slurm array job..."
JOB_ID=$(sbatch "${BASH_SOURCE[0]}" | awk '{print $NF}')
echo "Submitted array job: ${JOB_ID}"
echo "Monitor with: squeue -u ${USER}"
echo "Cancel with: scancel ${JOB_ID}"
