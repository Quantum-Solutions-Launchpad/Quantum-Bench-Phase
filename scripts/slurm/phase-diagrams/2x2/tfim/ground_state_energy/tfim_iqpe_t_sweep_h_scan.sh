#!/bin/bash
#SBATCH -J tfim-1d-E-h-iqpe-Lx8-t
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-t%a-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-t%a-%j.err

# TFIM 1D ground-state energy: h sweep for fixed t, Lx=8 (1D periodic boundary)
# Methods: IQPE
#
# t parameter: set via T_VALUE environment variable or SLURM_ARRAY_TASK_ID
# h parameter: 0.0 to 6.0 step 0.5
# System size: Lx = 8

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env

# Set t value from environment variable or array task ID
if [ -z "${T_VALUE:-}" ]; then
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        T_VALUE=$(printf "%.1f" $(echo "scale=1; ${SLURM_ARRAY_TASK_ID} / 10" | bc))
    else
        T_VALUE="${T_VALUE:-0.0}"
    fi
fi

DATA_DIR="${OUTPUT_ROOT}/logs/tfim/1d/ground-state-energy/h-sweep-Lx8"
PLOT_DIR="${OUTPUT_ROOT}/plots/tfim/1d/ground-state-energy/h-sweep-Lx8"

echo "==================================================================="
echo "TFIM 1D ground-state energy: h sweep (PBC, Lx=8, t=${T_VALUE})"
echo "  Method: IQPE"
echo "  Boundary: periodic"
echo "  System size: Lx = 8"
echo "  t value: ${T_VALUE}"
echo "  h range: 0.0 to 6.0 step 0.5"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command
cmd=(
    qbp run
    --method iqpe
    --observable E
    --qubit-operator /pscratch/sd/m/mbao202/tfim.hdf5
    --select "1D" --select "grid-pbc" --select "Lx-8"
    --x-param h
    --x-range 0.0 6.0 0.5
)

IQPE_TIME="${T_VALUE}" append_iqpe_args cmd

mkdir -p "${DATA_DIR}" "${PLOT_DIR}"

# Use t value in output filename
OUTPUT_TAG=$(printf "t%.1f" "${T_VALUE}")
cmd+=(
    --log-path "${DATA_DIR}/tfim-1d-pbc-Lx8-E-h-iqpe-${OUTPUT_TAG}.json"
    --plot-path "${PLOT_DIR}/tfim-1d-pbc-Lx8-E-h-iqpe-${OUTPUT_TAG}.pdf"
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
    echo "  Data: ${DATA_DIR}/tfim-1d-pbc-Lx8-E-h-iqpe-${OUTPUT_TAG}.json"
    echo "  Plot: ${PLOT_DIR}/tfim-1d-pbc-Lx8-E-h-iqpe-${OUTPUT_TAG}.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "ERROR: Computation failed with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
