#!/bin/bash
set -euo pipefail

setup_visualizer_env() {
    REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
    source "${REPO_ROOT}/scripts/slurm/phase-diagrams/qbp_sharded.sh"
    setup_qbp_slurm_env

    export LOKY_DISABLE_RESOURCE_TRACKER=1
    export JOBLIB_START_METHOD=spawn
    export LOKY_MAX_WORKERS=1
    export MPLBACKEND="${MPLBACKEND:-Agg}"

    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/manuscript-plots}"
    LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
    PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"
    PHI="${PHI:-0.7853981633974483}"

    mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${PLOT_DIR}" "${REPO_ROOT}/scripts/logs/slurm"
}

setup_visualizer_dmrg_env() {
    local default_threads="${1:-${SLURM_CPUS_PER_TASK:-1}}"

    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}-mpl}"
    export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/pscratch/sd/m/mbao202/julia_depot}"
    DMRG_THREADS="${DMRG_THREADS:-${default_threads}}"
    export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-${DMRG_THREADS}}"
    export OMP_NUM_THREADS="${DMRG_OMP_NUM_THREADS:-${DMRG_THREADS}}"
    export MKL_NUM_THREADS="${DMRG_MKL_NUM_THREADS:-${DMRG_THREADS}}"
    export OPENBLAS_NUM_THREADS="${DMRG_OPENBLAS_NUM_THREADS:-${DMRG_THREADS}}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}

append_qbp_output_paths() {
    local -n cmd_ref="$1"
    local log_path="$2"
    local plot_path="$3"

    mkdir -p "$(dirname "${log_path}")" "$(dirname "${plot_path}")"
    cmd_ref+=(
        --log-path "${log_path}"
        --plot-path "${plot_path}"
        --hide-plot
    )
}

append_vqe_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --vqe-iters "${VQE_ITERS:-10000}"
        --vqe-layers "${VQE_LAYERS:-5}"
        --vqe-reps "${VQE_REPS:-10}"
    )
}

append_iqpe_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --iqpe-time "${IQPE_TIME:-0.2}"
        --iqpe-trot "${IQPE_TROT:-5}"
        --iqpe-iters "${IQPE_ITERS:-8}"
        --iqpe-reps "${IQPE_REPS:-20}"
        --iqpe-initial-state vqe_informed
        --iqpe-initial-vqe-ansatz excitation_preserving
        --iqpe-initial-vqe-n-layers "${IQPE_INITIAL_VQE_N_LAYERS:-2}"
        --iqpe-initial-vqe-ansatz-kwarg "reps=${IQPE_INITIAL_VQE_REPS:-2}"
        --iqpe-initial-vqe-max-iters "${IQPE_INITIAL_VQE_MAX_ITERS:-1000}"
    )
}

append_dmrg_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --dmrg-nsweeps "${DMRG_NSWEEPS:-4}"
        --dmrg-maxdims "${DMRG_MAXDIMS:-20,50,100,200}"
        --dmrg-cutoff "${DMRG_CUTOFF:-1e-9}"
        --dmrg-seed "${DMRG_SEED:-1234}"
    )
}

run_visualizer_sharded_cmd() {
    local -n cmd_ref="$1"
    local cmd_string

    printf -v cmd_string "%q " "${cmd_ref[@]}"
    run_qbp_sharded_config "${cmd_string}"
}
