#!/bin/bash
#SBATCH -J haldane-2x2-band-noisy-analytic-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Run analytical and DMRG for band structure kx vs ky (noisy pipeline)
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/band_structure/haldane_band_structure_noisy_analytic_dmrg.sh
#
#   INTERACTIVE MODE (1 hour):
#     salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account m5027 \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/band_structure/haldane_band_structure_noisy_analytic.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env 16

DATA_DIR="${OUTPUT_ROOT}/logs/haldane/2x2/band-structure-noisy"
PLOT_DIR="${OUTPUT_ROOT}/plots/haldane/2x2/band-structure-noisy"

echo "==================================================================="
echo "Haldane 2x2 band structure kx vs ky (noisy)"
echo "  Methods: analytical, dmrg"
echo "  Backend: FakeSherbrooke"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command using helper functions
cmd=(
    qbp run
    --model haldane-honeycomb
    --method analytic
    --observable E
    --backend FakeSherbrooke
    --heatmap
    --x-param kx
    --x-range -3.141592653589793 3.141592653589793 0.39269908169872414
    --y-param ky
    --y-range -3.141592653589793 3.141592653589793 0.39269908169872414
    --t1 1.0
    --t2 0.1
    --phi 0.7853981633974483
    --M 0.0
)
append_qbp_output_paths cmd \
    "${DATA_DIR}/simulated-noisy-analytic-E-kx-vs-ky.json" \
    "${PLOT_DIR}/simulated-noisy-analytic-E-kx-vs-ky.pdf"

echo "Running computation..."
echo "Command: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo ""
    echo "==================================================================="
    echo "Computation complete!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/simulated-noisy-analytic-E-kx-vs-ky.json"
    echo "  Plot: ${PLOT_DIR}/simulated-noisy-analytic-E-kx-vs-ky.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Computation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
