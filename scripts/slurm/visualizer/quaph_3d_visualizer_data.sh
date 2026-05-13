#!/bin/bash
#SBATCH -J quaph-3d-data
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 6
#SBATCH -t 48:00:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/realspace_simulated.sh"
setup_realspace_env

export MPLBACKEND="${MPLBACKEND:-Agg}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/examples/logs}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/examples/plots}"
SIMULATION="${SIMULATION:-ideal}"

case "${SIMULATION}" in
    ideal|noisy) ;;
    *)
        echo "SIMULATION must be 'ideal' or 'noisy', got '${SIMULATION}'" >&2
        exit 1
        ;;
esac

MODELS=(haldane hubbard haldane-hubbard haldane hubbard haldane-hubbard)
N_SITES_VALUES=(4 4 4 6 6 6)
X_PARAMS=(n_occ n_occ t2 n_occ n_occ t2)
Y_PARAMS=(t2 U U t2 U U)

TOTAL_TASKS="${#MODELS[@]}"

mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

echo "Collecting QuAPH 3D visualizer data"
echo "  job id:      ${SLURM_JOB_ID:-manual}"
echo "  simulation:  ${SIMULATION}"
echo "  configs:     ${TOTAL_TASKS}"
echo "  nodes:       ${SLURM_JOB_NUM_NODES:-local}"
echo "  shards:      ${SHARDS}"
echo "  log dir:     ${LOG_DIR}"
echo "  plot dir:    ${PLOT_DIR}"

run_visualizer_config() {
    local task_id="$1"
    local model="${MODELS[$task_id]}"
    local n_sites="${N_SITES_VALUES[$task_id]}"
    local x_param="${X_PARAMS[$task_id]}"
    local y_param="${Y_PARAMS[$task_id]}"

    local cmd=(
        "simulated-${SIMULATION}"
        --model "${model}"
        --n-sites "${n_sites}"
        --x-param "${x_param}"
        --y-param "${y_param}"
        --log-dir "${LOG_DIR}"
        --plot-dir "${PLOT_DIR}"
        --hide-plot
        --hide-legend
    )

    if [[ "${model}" == "haldane" ]]; then
        cmd+=(--y-range "${HALDANE_T2_START:-0.0}" "${HALDANE_T2_END:-1.0}" "${HALDANE_T2_STEP:-0.1}")
    elif [[ "${model}" == "hubbard" ]]; then
        cmd+=(--y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-4.0}" "${HUBBARD_U_STEP:-0.5}")
    elif [[ "${model}" == "haldane-hubbard" ]]; then
        cmd+=(--x-range "${HALDANE_HUBBARD_T2_START:-0.0}" "${HALDANE_HUBBARD_T2_END:-1.5}" "${HALDANE_HUBBARD_T2_STEP:-0.1}")
        cmd+=(--y-range "${HALDANE_HUBBARD_U_START:-0.0}" "${HALDANE_HUBBARD_U_END:-4.0}" "${HALDANE_HUBBARD_U_STEP:-0.5}")
    fi

    if (( ${VQE_REPS:-10} > 0 )); then
        cmd+=(--vqe-iters "${VQE_ITERS:-10000}")
        cmd+=(--vqe-layers "${VQE_LAYERS:-5}")
        cmd+=(--vqe-reps "${VQE_REPS:-10}")
    else
        cmd+=(--vqe-reps 0)
    fi

    if (( ${IQPE_REPS:-20} > 0 )); then
        cmd+=(--iqpe-time "${IQPE_TIME:-0.2}")
        cmd+=(--iqpe-trot "${IQPE_TROT:-5}")
        cmd+=(--iqpe-iters "${IQPE_ITERS:-8}")
        cmd+=(--iqpe-reps "${IQPE_REPS:-20}")
    else
        cmd+=(--iqpe-reps 0)
    fi

    echo "Starting config ${task_id}/${TOTAL_TASKS}: ${model}, ${n_sites}-sites, ${x_param} vs ${y_param}"
    printf -v cmd_string "%q " "${cmd[@]}"
    run_sharded_config "${cmd_string}"

    local summary="${LOG_DIR}/${model}/${n_sites}-sites/simulated-${SIMULATION}-${x_param}-vs-${y_param}.json"
    local raw="${LOG_DIR}/${model}/${n_sites}-sites/raw-data/simulated-${SIMULATION}-${x_param}-vs-${y_param}.json"
    local plot="${PLOT_DIR}/${model}/${n_sites}-sites/simulated-${SIMULATION}-${x_param}-vs-${y_param}.pdf"
    echo "Completed config ${task_id} at $(date)"
    echo "  summary: ${summary}"
    echo "  raw:     ${raw}"
    echo "  plot:    ${plot}"
}

for task_id in $(seq 0 $((TOTAL_TASKS - 1))); do
    run_visualizer_config "${task_id}" &
done
wait

echo "All configurations completed at $(date)"
