#!/bin/bash
#SBATCH -J haldane-2x2-ideal
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
source "${REPO_ROOT}/scripts/slurm/visualizer/common_visualizer.sh"
setup_visualizer_env

cmd=(
    simulated-ideal
)
append_haldane_2x2_phase_args cmd
append_vqe_iqpe_args cmd
append_output_args cmd

print_visualizer_header "Running Haldane 2x2 simulated-ideal phase diagram"
echo "  t2 sweep:    ${HALDANE_T2_START:-0.0} ${HALDANE_T2_END:-1.0} ${HALDANE_T2_STEP:-0.1}"

run_visualizer_sharded_cmd cmd

echo "Completed Haldane 2x2 run at $(date)"
echo "  summary: ${LOG_DIR}/haldane/2x2/simulated-ideal-3d-n_occ-vs-t2.json"
echo "  raw:     ${LOG_DIR}/haldane/2x2/raw-data/simulated-ideal-3d-n_occ-vs-t2.json"
echo "  plot:    ${PLOT_DIR}/haldane/2x2/simulated-ideal-3d-n_occ-vs-t2.pdf"
