#!/bin/bash
#SBATCH -J qbp-compare-data
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 6
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH --array=0-5%2
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-8}"

ALGORITHMS="${ALGORITHMS:-exact vqe iqpe dmrg}"

MODELS=(haldane hubbard haldane-hubbard haldane hubbard haldane-hubbard)
LATTICES=("2 2" "2 2" "2 2" "3 3" "3 3" "3 3")
X_PARAMS=(n_occ n_occ t2 n_occ n_occ t2)
Y_PARAMS=(t2 U U t2 U U)

TOTAL_TASKS="${#MODELS[@]}"
TASK_IDS="${TASK_IDS:-}"

print_visualizer_header "Collecting QuAPH comparison visualizer data"
echo "  algorithms:  ${ALGORITHMS}"
echo "  quantum:     ${QUANTUM_PIPELINE:-ideal}"
echo "  configs:     ${TOTAL_TASKS}"
echo "  dmrg:        nsweeps=${DMRG_NSWEEPS:-4}, maxdims=${DMRG_MAXDIMS:-20,50,100,200}, cutoff=${DMRG_CUTOFF:-1e-9}"
echo "  threads:     ${DMRG_THREADS} per shard"
if [[ -n "${TASK_IDS}" ]]; then
    echo "  task ids:    ${TASK_IDS}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "  array task:  ${SLURM_ARRAY_TASK_ID}"
fi

run_visualizer_config() {
    local task_id="$1"
    local model="${MODELS[$task_id]}"
    local lattice="${LATTICES[$task_id]}"
    local x_param="${X_PARAMS[$task_id]}"
    local y_param="${Y_PARAMS[$task_id]}"
    local algorithms=()
    read -r -a algorithms <<< "${ALGORITHMS}"

    local cmd=(
        compare
        --model "${model}"
        --lattice ${lattice}
        --x-param "${x_param}"
        --y-param "${y_param}"
        --algorithms "${algorithms[@]}"
        --quantum-pipeline "${QUANTUM_PIPELINE:-ideal}"
    )

    if [[ "${model}" == "haldane" ]]; then
        cmd+=(--t1 "${HALDANE_T1:-1.0}" --phi "${HALDANE_PHI:-${PHI}}" --M "${HALDANE_M:-0.0}")
        cmd+=(--y-range "${HALDANE_T2_START:-0.0}" "${HALDANE_T2_END:-1.0}" "${HALDANE_T2_STEP:-0.1}")
    elif [[ "${model}" == "hubbard" ]]; then
        cmd+=(--t "${HUBBARD_T:-1.0}")
        cmd+=(--y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-4.0}" "${HUBBARD_U_STEP:-0.5}")
    elif [[ "${model}" == "haldane-hubbard" ]]; then
        cmd+=(--t1 "${HALDANE_HUBBARD_T1:-1.0}" --phi "${HALDANE_HUBBARD_PHI:-${PHI}}" --M "${HALDANE_HUBBARD_M:-0.0}")
        cmd+=(--x-range "${HALDANE_HUBBARD_T2_START:-0.0}" "${HALDANE_HUBBARD_T2_END:-1.5}" "${HALDANE_HUBBARD_T2_STEP:-0.1}")
        cmd+=(--y-range "${HALDANE_HUBBARD_U_START:-0.0}" "${HALDANE_HUBBARD_U_END:-4.0}" "${HALDANE_HUBBARD_U_STEP:-0.5}")
    fi

    append_optional_vqe_iqpe_args cmd
    append_compare_dmrg_args cmd
    append_output_args cmd

    echo "Starting config ${task_id}/${TOTAL_TASKS}: ${model}, lattice ${lattice}, ${x_param} vs ${y_param}"
    run_visualizer_sharded_cmd cmd

    local lattice_tag="${lattice// /x}"
    local compare_tag="compare-${x_param}-vs-${y_param}"
    if [[ "${QUANTUM_PIPELINE:-ideal}" != "ideal" ]]; then
        compare_tag="compare-${QUANTUM_PIPELINE}-${x_param}-vs-${y_param}"
    fi
    local summary="${LOG_DIR}/${model}/${lattice_tag}/${compare_tag}.json"
    local plot="${PLOT_DIR}/${model}/${lattice_tag}/${compare_tag}.pdf"
    echo "Completed config ${task_id} at $(date)"
    echo "  summary: ${summary}"
    echo "  plot:    ${plot}"
}

if [[ -n "${TASK_IDS}" ]]; then
    task_ids="${TASK_IDS}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_ids="${SLURM_ARRAY_TASK_ID}"
else
    task_ids="$(seq 0 $((TOTAL_TASKS - 1)))"
fi

for task_id in ${task_ids}; do
    if (( task_id < 0 || task_id >= TOTAL_TASKS )); then
        echo "Invalid task id ${task_id}; valid range is 0-$((TOTAL_TASKS - 1))" >&2
        exit 1
    fi
    run_visualizer_config "${task_id}"
done

echo "All configurations completed at $(date)"
