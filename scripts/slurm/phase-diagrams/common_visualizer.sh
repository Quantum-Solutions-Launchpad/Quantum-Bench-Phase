#!/bin/bash
set -euo pipefail

setup_visualizer_env() {
    REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
    source "${REPO_ROOT}/scripts/slurm/phase-diagrams/quaph_sharded.sh"
    setup_quaph_slurm_env

    export MPLBACKEND="${MPLBACKEND:-Agg}"

    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/examples/logs}"
    PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/examples/plots}"
    PHI="${PHI:-0.7853981633974483}"

    mkdir -p "${LOG_DIR}" "${PLOT_DIR}" "${REPO_ROOT}/scripts/logs/slurm"
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

print_visualizer_header() {
    local title="$1"

    echo "${title}"
    echo "  job id:      ${SLURM_JOB_ID:-manual}"
    echo "  nodes:       ${SLURM_JOB_NUM_NODES:-local}"
    echo "  shards:      ${SHARDS}"
    echo "  log dir:     ${LOG_DIR}"
    echo "  plot dir:    ${PLOT_DIR}"
}

append_haldane_2x2_phase_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --model haldane
        --lattice 2 2
        --x-param n_occ
        --y-param t2
        --y-range "${HALDANE_T2_START:-0.0}" "${HALDANE_T2_END:-1.0}" "${HALDANE_T2_STEP:-0.1}"
        --t1 "${HALDANE_T1:-1.0}"
        --phi "${HALDANE_PHI:-${PHI}}"
        --M "${HALDANE_M:-0.0}"
    )
}

append_output_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --log-dir "${LOG_DIR}"
        --plot-dir "${PLOT_DIR}"
        --hide-plot
    )
}

append_vqe_iqpe_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --vqe-iters "${VQE_ITERS:-10000}"
        --vqe-layers "${VQE_LAYERS:-5}"
        --vqe-reps "${VQE_REPS:-10}"
        --iqpe-time "${IQPE_TIME:-0.2}"
        --iqpe-trot "${IQPE_TROT:-5}"
        --iqpe-iters "${IQPE_ITERS:-8}"
        --iqpe-reps "${IQPE_REPS:-20}"
    )
}

append_optional_vqe_iqpe_args() {
    local -n cmd_ref="$1"

    if (( ${VQE_REPS:-10} > 0 )); then
        cmd_ref+=(--vqe-iters "${VQE_ITERS:-10000}")
        cmd_ref+=(--vqe-layers "${VQE_LAYERS:-5}")
        cmd_ref+=(--vqe-reps "${VQE_REPS:-10}")
    else
        cmd_ref+=(--vqe-reps 0)
    fi

    if (( ${IQPE_REPS:-20} > 0 )); then
        cmd_ref+=(--iqpe-time "${IQPE_TIME:-0.2}")
        cmd_ref+=(--iqpe-trot "${IQPE_TROT:-5}")
        cmd_ref+=(--iqpe-iters "${IQPE_ITERS:-8}")
        cmd_ref+=(--iqpe-reps "${IQPE_REPS:-20}")
    else
        cmd_ref+=(--iqpe-reps 0)
    fi
}

append_dmrg_run_args() {
    local -n cmd_ref="$1"

    cmd_ref+=(
        --nsweeps "${DMRG_NSWEEPS:-4}"
        --maxdims "${DMRG_MAXDIMS:-20,50,100,200}"
        --cutoff "${DMRG_CUTOFF:-1e-9}"
        --seed "${DMRG_SEED:-1234}"
    )
}

append_compare_dmrg_args() {
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
    run_quaph_sharded_config "${cmd_string}"
}
