#!/bin/bash
#SBATCH -J haldane-2x2-band-noisy-dmrg
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

# Haldane 2x2 band structure kx vs ky sweep (noisy pipeline) - DMRG only
# Runs DMRG method separately for band structure visualization
#
# Usage:
#   BATCH MODE:
#     sbatch /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/band_structure/haldane_band_structure_noisy_dmrg.sh
#
#   INTERACTIVE MODE (4 hours):
#     salloc -C cpu -N 1 --ntasks-per-node=4 -c 8 -t 4:00:00 -A m5027 -q regular \
#     bash /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/2x2/haldane/band_structure/haldane_band_structure_noisy_dmrg.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
SHARDS="${SHARDS:-${SLURM_NTASKS:-1}}"
setup_visualizer_env
setup_visualizer_dmrg_env 1
export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-8}"

DATA_DIR="${LOG_DIR}/haldane/2x2/band-structure-noisy"
PLOT_DIR="${PLOT_DIR}/haldane/2x2/band-structure-noisy"

X_PARAM=kx
X_START="${HALDANE_KX_START:--3.141592653589793}"
X_END="${HALDANE_KX_END:-3.141592653589793}"
X_STEP="${HALDANE_KX_STEP:-0.39269908169872414}"
Y_PARAM=ky
Y_START="${HALDANE_KY_START:--3.141592653589793}"
Y_END="${HALDANE_KY_END:-3.141592653589793}"
Y_STEP="${HALDANE_KY_STEP:-0.39269908169872414}"
SWEEP_TAG="kx-vs-ky"

PIPELINE="${PIPELINE:-noisy}"
backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
fixed_args=(
    --t1 "${HALDANE_T1:-1.0}"
    --t2 "${HALDANE_T2:-0.1}"
    --phi "${HALDANE_PHI:-0.7853981633974483}"
    --M "${HALDANE_M:-0.0}"
)

echo "==================================================================="
echo "Haldane 2x2 band structure kx vs ky (noisy) - DMRG"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  backend:     ${NOISY_BACKEND:-FakeSherbrooke}"
echo "  methods:     dmrg"
echo "  grid points: ~289 (17 x 17)"
echo "  parallelism: distributed across ${SHARDS} shard(s)"
echo "  cell jobs:   ${QBP_JOBS_PER_SHARD} per shard"
echo "  output:      ${OUTPUT_ROOT}"
echo "==================================================================="

# Build command
cmd=(
    --model haldane
    --method dmrg
    --observable E
    --heatmap
    --x-param "${X_PARAM}"
    --x-range "${X_START}" "${X_END}" "${X_STEP}"
    --y-param "${Y_PARAM}"
    --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
    "${fixed_args[@]}"
    "${backend_args[@]}"
    --dmrg-nsweeps "${DMRG_NSWEEPS:-1}"
    --dmrg-maxdims "${DMRG_MAXDIMS:-20,50,100,200,400}"
    --dmrg-cutoff "${DMRG_CUTOFF:-1e-10}"
)

mkdir -p "${DATA_DIR}" "${PLOT_DIR}"

append_qbp_output_paths cmd "${DATA_DIR}/simulated-${PIPELINE}-dmrg-E-${SWEEP_TAG}.json" "${PLOT_DIR}/simulated-${PIPELINE}-dmrg-E-${SWEEP_TAG}.pdf"

if run_visualizer_sharded_cmd cmd; then
    echo ""
    echo "==================================================================="
    echo "Computation complete!"
    echo "  Finished: $(date)"
    echo "Output saved to:"
    echo "  Data: ${DATA_DIR}/simulated-${PIPELINE}-dmrg-E-${SWEEP_TAG}.json"
    echo "  Plot: ${PLOT_DIR}/simulated-${PIPELINE}-dmrg-E-${SWEEP_TAG}.pdf"
    echo "==================================================================="
else
    EXIT_STATUS=$?
    echo ""
    echo "==================================================================="
    echo "ERROR: Computation failed with exit status $EXIT_STATUS"
    echo "==================================================================="
    exit $EXIT_STATUS
fi
