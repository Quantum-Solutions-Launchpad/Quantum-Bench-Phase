#!/bin/bash
#SBATCH -J haldane-2x2-E-M-phi-analytic-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Rerun analytical and DMRG for M vs phi with exact same parameters
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/M_vs_phi_analytical_dmrg.sh
#
#   INTERACTIVE MODE (1 hour):
#     salloc -A m5027 -q regular -C cpu -N 1 --ntasks-per-node=8 -c 16 --qos interactive -t 01:00:00  \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/M_vs_phi_analytical.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env

DATA_DIR="${OUTPUT_ROOT}/logs/haldane/2x2/ground-state-energy/M-vs-phi"
PLOT_DIR="${OUTPUT_ROOT}/plots/haldane/2x2/ground-state-energy/M-vs-phi"

echo "==================================================================="
echo "Rerunning Haldane 2x2 ground-state energy M vs phi"
echo "  Methods: analytic"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command using helper functions
cmd=(
    qbp run
    --model haldane
    --method analytic
    --lattice 2 2
    --observable E
    --x-param phi
    --x-range -3.141592653589793 3.141592653589793 0.78539816
    --y-param M
    --y-range -6.0 6.0 1
    --t1 1.0
    --t2 1.0
)

append_qbp_output_paths cmd \
    "${DATA_DIR}/simulated-ideal-analytic-E-M-vs-phi.json" \
    "${PLOT_DIR}/simulated-ideal-analytic-E-M-vs-phi.pdf"

echo "Running recomputation..."
echo "Command: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo ""
    echo "==================================================================="
    echo "Recomputation complete!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/simulated-ideal-analytic-E-M-vs-phi.json"
    echo "  Plot: ${PLOT_DIR}/simulated-ideal-analytic-E-M-vs-phi.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Recomputation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
