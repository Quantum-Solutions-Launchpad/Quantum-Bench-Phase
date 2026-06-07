#!/bin/bash
#SBATCH -J haldane-2x2-compare
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-8}"

print_visualizer_header "Running Haldane 2x2 exact/VQE/IQPE/DMRG comparison phase diagram"
echo "  t2 sweep:    ${HALDANE_T2_START:-0.0} ${HALDANE_T2_END:-1.0} ${HALDANE_T2_STEP:-0.1}"
echo "  quantum:     ${QUANTUM_PIPELINE:-ideal}"
echo "  vqe:         reps=${VQE_REPS:-10}, iters=${VQE_ITERS:-10000}, layers=${VQE_LAYERS:-5}"
echo "  iqpe:        reps=${IQPE_REPS:-20}, time=${IQPE_TIME:-0.2}, trot=${IQPE_TROT:-5}, iters=${IQPE_ITERS:-8}"
echo "  dmrg:        nsweeps=${DMRG_NSWEEPS:-4}, maxdims=${DMRG_MAXDIMS:-20,50,100,200}, cutoff=${DMRG_CUTOFF:-1e-9}"
echo "  threads:     ${DMRG_THREADS} per shard"

cmd=(
    compare
    --algorithms exact vqe iqpe dmrg
    --quantum-pipeline "${QUANTUM_PIPELINE:-ideal}"
)
append_haldane_2x2_phase_args cmd
append_vqe_iqpe_args cmd
append_compare_dmrg_args cmd
append_output_args cmd

run_visualizer_sharded_cmd cmd

COMPARE_TAG="compare-n_occ-vs-t2"
if [[ "${QUANTUM_PIPELINE:-ideal}" != "ideal" ]]; then
    COMPARE_TAG="compare-${QUANTUM_PIPELINE}-n_occ-vs-t2"
fi

echo "Completed Haldane 2x2 comparison run at $(date)"
echo "  summary: ${LOG_DIR}/haldane/2x2/${COMPARE_TAG}.json"
echo "  plot:    ${PLOT_DIR}/haldane/2x2/${COMPARE_TAG}.pdf"
