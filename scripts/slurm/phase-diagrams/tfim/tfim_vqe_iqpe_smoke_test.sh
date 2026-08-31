#!/bin/bash
#SBATCH -J tfim-vqe-iqpe-smoke-test
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 0:30:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# TFIM VQE/IQPE smoke test
# Quick test to verify VQE and IQPE work on TFIM hamlib data
# Runs only Lx=4,5 and h=0,1 (minimal sweep)
#
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/tfim/tfim_vqe_iqpe_smoke_test.sh
#
#   INTERACTIVE MODE (30 min):
#     salloc -C cpu -N 1 --ntasks-per-node=1 -c 32 -t 0:30:00 -A m5027 -q regular \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/tfim/tfim_vqe_iqpe_smoke_test.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env

DATA_DIR="${OUTPUT_ROOT}/logs/tfim/1d/smoke-test"
PLOT_DIR="${OUTPUT_ROOT}/plots/tfim/1d/smoke-test"

echo "==================================================================="
echo "TFIM VQE/IQPE Smoke Test"
echo "  Methods: VQE, IQPE"
echo "  Boundary: PBC"
echo "  Sweep: Lx=[4,5], h=[0.0, 1.0]"
echo "  Data dir: ${DATA_DIR}"
echo "  Started: $(date)"
echo "==================================================================="

# Build command - minimal sweep for quick test
cmd=(
    qbp run
    --method vqe iqpe
    --observable E
    --qubit-operator /pscratch/sd/m/mbao202/tfim.hdf5
    --select "1D" --select "grid-pbc"
    --x-param Lx
    --x-range 4 5 1
    --y-param h
    --y-range 0.0 1.0 1.0
)

# Minimal parameters - just verify pipeline works
cmd+=(
    --vqe-iters 50
    --vqe-layers 1
    --vqe-reps 1
    --iqpe-time 0.05
    --iqpe-trot 1
    --iqpe-iters 1
    --iqpe-reps 1
    --iqpe-initial-state vqe_informed
    --iqpe-initial-vqe-ansatz excitation_preserving
    --iqpe-initial-vqe-n-layers 1
    --iqpe-initial-vqe-max-iters 100
)

mkdir -p "${DATA_DIR}" "${PLOT_DIR}"

cmd+=(
    --log-path "${DATA_DIR}/tfim-vqe-iqpe-smoke-test.json"
    --plot-path "${PLOT_DIR}/tfim-vqe-iqpe-smoke-test.pdf"
)

echo "Running computation..."
echo "Command: ${cmd[@]}"
echo ""

if "${cmd[@]}"; then
    echo ""
    echo "==================================================================="
    echo "✅ Smoke test PASSED!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/tfim-vqe-iqpe-smoke-test.json"
    echo "  Plot: ${PLOT_DIR}/tfim-vqe-iqpe-smoke-test.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "❌ Smoke test FAILED with status $EXIT_STATUS"
    echo "Failed at: $(date)"
    exit $EXIT_STATUS
fi
