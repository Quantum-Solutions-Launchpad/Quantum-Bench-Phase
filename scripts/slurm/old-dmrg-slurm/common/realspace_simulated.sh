#!/bin/bash
set -euo pipefail

setup_realspace_env() {
    REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
    cd "${REPO_ROOT}" || exit 1

    module load python
    source "${REPO_ROOT}/venv/bin/activate"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}-mpl}"

    SHARDS="${SHARDS:-${SLURM_NTASKS:-${SLURM_CPUS_ON_NODE:-128}}}"
    CPUS_PER_SHARD="${CPUS_PER_SHARD:-${SLURM_CPUS_PER_TASK:-1}}"
    export QUAPH_JOBS_PER_SHARD="${QUAPH_JOBS_PER_SHARD:-1}"
}

run_command() {
    local cmd="$1"
    shift

    if [[ "${cmd}" == scripts/* || "${cmd}" == *.py* ]]; then
        python ${cmd} "$@"
    else
        quaph run ${cmd} "$@"
    fi
}

run_sharded_config() {
    local cmd="$1"
    local pids=()
    local status=0

    run_command "${cmd}" --prepare-only
    for shard in $(seq 0 $((SHARDS - 1))); do
        if [[ "${cmd}" == scripts/* || "${cmd}" == *.py* ]]; then
            srun -N 1 -n 1 -c "${CPUS_PER_SHARD}" --exact bash -c "python ${cmd} --task-index ${shard} --task-count ${SHARDS}" &
        else
            srun -N 1 -n 1 -c "${CPUS_PER_SHARD}" --exact bash -c "quaph run ${cmd} --task-index ${shard} --task-count ${SHARDS}" &
        fi
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    if (( status != 0 )); then
        echo "One or more shards failed; skipping aggregation." >&2
        return "${status}"
    fi
    run_command "${cmd}" --aggregate-only
}

run_single_config() {
    local cmd="$1"
    run_sharded_config "${cmd}"
}
