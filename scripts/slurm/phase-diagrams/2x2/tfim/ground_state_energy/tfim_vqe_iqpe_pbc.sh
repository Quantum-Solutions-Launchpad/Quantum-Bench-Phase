#!/bin/bash
#SBATCH -J tfim-1d-E-Lx-h-vqe-iqpe
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 8:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# TFIM ground-state energy: Lx vs h (1D periodic boundary)
# Methods: VQE + IQPE
#
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/tfim/ground_state_energy/tfim_vqe_iqpe.sh
#
#   INTERACTIVE MODE (8 hours):
#     salloc -C cpu -N 1 --ntasks-per-node=1 -c 32 -t 8:00:00 -A m5027 -q regular \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/tfim/ground_state_energy/tfim_vqe_iqpe.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env

DATA_DIR="${OUTPUT_ROOT}/logs/tfim/1d/ground-state-energy/Lx-vs-h-pbc"
PLOT_DIR="${OUTPUT_ROOT}/plots/tfim/1d/ground-state-energy/Lx-vs-h-pbc"

echo "==================================================================="
echo "TFIM 1D ground-state energy Lx vs h (PBC)"
echo "  Methods: VQE, IQPE"
echo "  Boundary: periodic"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command
cmd=(
    qbp run
    --method vqe iqpe
    --observable E
    --qubit-operator /pscratch/sd/m/mbao202/tfim.hdf5
    --select "1D" --select "grid-pbc"
    --x-param Lx
    --x-range 4 12 1
    --y-param h
    --y-range 0.0 6.0 0.5
)

append_vqe_args cmd
append_iqpe_args cmd
cmd+=(--iqpe-initial-state computational_zero)

mkdir -p "${DATA_DIR}" "${PLOT_DIR}"

cmd+=(
    --log-path "${DATA_DIR}/tfim-1d-pbc-E-Lx-vs-h-vqe-iqpe.json"
    --plot-path "${PLOT_DIR}/tfim-1d-pbc-E-Lx-vs-h-vqe-iqpe.pdf"
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
    echo "  Data: ${DATA_DIR}/tfim-1d-pbc-E-Lx-vs-h-vqe-iqpe.json"
    echo "  Plot: ${PLOT_DIR}/tfim-1d-pbc-E-Lx-vs-h-vqe-iqpe.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Computation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
