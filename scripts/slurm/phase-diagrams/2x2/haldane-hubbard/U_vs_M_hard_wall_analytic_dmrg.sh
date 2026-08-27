#!/bin/bash
#SBATCH -J hh-2x2-E-M-U-hw-analytic-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Run analytical and DMRG for M vs U (hard wall boundary)
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane-hubbard/U_vs_M_hard_wall_analytic_dmrg.sh
#
#   INTERACTIVE MODE (1 hour):
#     salloc -C cpu -N 2 --ntasks-per-node=8 -c 16 -t 1:00:00 -A m5027 -q regular \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane-hubbard/U_vs_M_hard_wall_analytic_dmrg.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env 16

PIPELINE="noisy"
backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")

DATA_DIR="${OUTPUT_ROOT}/logs/haldane/2x2/ground-state-energy/M-vs-U-hard-wall"
PLOT_DIR="${OUTPUT_ROOT}/plots/haldane/2x2/ground-state-energy/M-vs-U-hard-wall"

echo "==================================================================="
echo "Haldane-Hubbard 2x2 ground-state energy M vs U (hard wall)"
echo "  Methods: analytical, dmrg"
echo "  Boundary: open (hard wall)"
echo "  Pipeline: ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command using helper functions
cmd=(
    qbp run
    --model haldane-hubbard-honeycomb
    --method analytic dmrg
    --lattice 2 2
    --observable E
    --boundary open
    --x-param M
    --x-range 0 3 0.25
    --y-param U
    --y-range 0.0 3.0 0.25
    --t1 1.0
    --t2 1.0
    --phi 0.7853981633974483
    --n-occ 4
)

cmd+=("${backend_args[@]}")

append_dmrg_args cmd
append_qbp_output_paths cmd \
    "${DATA_DIR}/simulated-${PIPELINE}-analytic-dmrg-E-M-vs-U-hard-wall.json" \
    "${PLOT_DIR}/simulated-${PIPELINE}-analytic-dmrg-E-M-vs-U-hard-wall.pdf"

echo "Running computation..."
echo "Command: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo ""
    echo "==================================================================="
    echo "Computation complete!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/simulated-${PIPELINE}-analytic-dmrg-E-M-vs-U-hard-wall.json"
    echo "  Plot: ${PLOT_DIR}/simulated-${PIPELINE}-analytic-dmrg-E-M-vs-U-hard-wall.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Computation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
