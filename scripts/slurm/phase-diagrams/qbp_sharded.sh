#!/bin/bash
set -euo pipefail

setup_qbp_slurm_env() {
    REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
    cd "${REPO_ROOT}" || exit 1

    module load python
    source "${REPO_ROOT}/venv/bin/activate"

    QBP_CLI="${QBP_CLI:-qbp}"
    if ! command -v "${QBP_CLI}" >/dev/null 2>&1; then
        echo "ERROR: qbp command not found. Did you activate the venv?" >&2
        exit 1
    fi
    export QBP_CLI

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}-mpl}"

    SHARDS="${SHARDS:-${SLURM_NTASKS:-${SLURM_CPUS_ON_NODE:-128}}}"
    CPUS_PER_SHARD="${CPUS_PER_SHARD:-${SLURM_CPUS_PER_TASK:-1}}"
    export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-${CPUS_PER_SHARD}}"
}

run_qbp_sharded_config() {
    local cmd="$1"
    local pids=()
    local status=0

    # Skip prepare-only: each shard will compute its own data, avoiding
    # sequential bottleneck of computing analytic results on all grid points
    # in a single process before parallel execution.

    for shard in $(seq 0 $((SHARDS - 1))); do
        srun -N 1 -n 1 -c "${CPUS_PER_SHARD}" --exact \
            bash -c "${QBP_CLI} run ${cmd} --task-index ${shard} --task-count ${SHARDS}" &
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
    ${QBP_CLI} run ${cmd} --aggregate-only
}
