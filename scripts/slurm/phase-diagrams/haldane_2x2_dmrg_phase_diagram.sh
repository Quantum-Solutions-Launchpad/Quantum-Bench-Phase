#!/bin/bash
#SBATCH -J haldane-2x2-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-16}"

print_visualizer_header "Running Haldane 2x2 ITensorMPS DMRG phase diagram against exact values"
echo "  t2 sweep:    ${HALDANE_T2_START:-0.0} ${HALDANE_T2_END:-1.0} ${HALDANE_T2_STEP:-0.1}"
echo "  nsweeps:     ${DMRG_NSWEEPS:-4}"
echo "  maxdims:     ${DMRG_MAXDIMS:-20,50,100,200}"
echo "  cutoff:      ${DMRG_CUTOFF:-1e-9}"
echo "  threads:     ${DMRG_THREADS} per shard"

cmd=(
    dmrg
)
append_haldane_2x2_phase_args cmd
append_output_args cmd
append_dmrg_run_args cmd

run_visualizer_sharded_cmd cmd

echo "Completed Haldane 2x2 DMRG run at $(date)"
echo "  summary: ${LOG_DIR}/haldane/2x2/dmrg/dmrg-n_occ-vs-t2.json"
echo "  raw dir: ${LOG_DIR}/haldane/2x2/dmrg/raw-data"
echo "  plot:    ${PLOT_DIR}/haldane/2x2/dmrg/dmrg-n_occ-vs-t2.pdf"
