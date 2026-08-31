#!/bin/bash
#SBATCH -J tfim-1d-E-Lx-h-dmrg-pbc
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 4:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# TFIM ground-state energy: Lx vs h (1D periodic boundary)
# Methods: DMRG (tensor network)
#
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/tfim/ground_state_energy/tfim_dmrg_pbc.sh
#
#   INTERACTIVE MODE (1 hours):
#     salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account m5027 \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/tfim/ground_state_energy/tfim_dmrg_pbc.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env 1

DATA_DIR="${OUTPUT_ROOT}/logs/tfim/1d/ground-state-energy/Lx-vs-h-pbc"
PLOT_DIR="${OUTPUT_ROOT}/plots/tfim/1d/ground-state-energy/Lx-vs-h-pbc"

echo "==================================================================="
echo "TFIM 1D ground-state energy Lx vs h (PBC)"
echo "  Methods: DMRG"
echo "  Boundary: periodic"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command
cmd=(
    qbp run
    --method dmrg
    --observable E
    --qubit-operator /pscratch/sd/m/mbao202/tfim.hdf5
    --select "1D" --select "grid-pbc"
    --x-param Lx
    --x-range 4 12 1
    --y-param h
    --y-range 0.0 6.0 0.5
)
append_dmrg_args cmd
mkdir -p "${DATA_DIR}" "${PLOT_DIR}"

cmd+=(
    --log-path "${DATA_DIR}/tfim-1d-pbc-E-Lx-vs-h-dmrg.json"
    --plot-path "${PLOT_DIR}/tfim-1d-pbc-E-Lx-vs-h-dmrg.pdf"
)

echo "Running computation..."
echo "Command: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo ""
    echo "==================================================================="
    echo "Computation complete!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/tfim-1d-pbc-E-Lx-vs-h-dmrg.json"
    echo "  Plot: ${PLOT_DIR}/tfim-1d-pbc-E-Lx-vs-h-dmrg.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Computation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
