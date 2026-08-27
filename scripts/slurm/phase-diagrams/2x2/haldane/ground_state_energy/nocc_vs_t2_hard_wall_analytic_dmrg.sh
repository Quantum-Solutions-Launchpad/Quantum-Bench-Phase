#!/bin/bash
#SBATCH -J haldane-2x2-E-nocc-t2-hard-wall-analytic-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Run analytical and DMRG for n_occ vs t2 (hard wall boundary)
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/nocc_vs_t2_hard_wall_analytic_dmrg.sh
#
#   INTERACTIVE MODE (1 hour):
#     salloc -C cpu -N 2 --ntasks-per-node=8 -c 16 -t 1:00:00 -A m5027 -q regular \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/ground_state_energy/nocc_vs_t2_hard_wall_analytic_dmrg.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env 16

DATA_DIR="${OUTPUT_ROOT}/logs/haldane/2x2/ground-state-energy/nocc-vs-t2-hard-wall"
PLOT_DIR="${OUTPUT_ROOT}/plots/haldane/2x2/ground-state-energy/nocc-vs-t2-hard-wall"

echo "==================================================================="
echo "Haldane 2x2 ground-state energy n_occ vs t2 (hard wall)"
echo "  Methods: analytical, dmrg"
echo "  Boundary: open (hard wall)"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command using helper functions
cmd=(
    qbp run
    --model haldane-honeycomb
    --method analytic dmrg
    --lattice 2 2
    --observable E
    --boundary open
    --x-param n_occ
    --x-range 0 8 1
    --y-param t2
    --y-range 0.0 1.0 0.1
    --t1 1.0
    --phi 0.7853981633974483
    --M 0.0
)

append_dmrg_args cmd
append_qbp_output_paths cmd \
    "${DATA_DIR}/simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json" \
    "${PLOT_DIR}/simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.pdf"

echo "Running computation..."
echo "Command: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo ""
    echo "==================================================================="
    echo "Computation complete!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json"
    echo "  Plot: ${PLOT_DIR}/simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Computation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
