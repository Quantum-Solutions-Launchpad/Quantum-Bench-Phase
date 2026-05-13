#!/bin/bash

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

    SHARDS="${SHARDS:-${SLURM_CPUS_ON_NODE:-128}}"
}

run_sharded_config() {
    local cmd="$1"

    quaph run ${cmd} --prepare-only
    for shard in $(seq 0 $((SHARDS - 1))); do
        srun -N 1 -n 1 -c 1 --exact bash -c "quaph run ${cmd} --task-index ${shard} --task-count ${SHARDS}" &
    done
    wait
    quaph run ${cmd} --aggregate-only
}

run_single_config() {
    local cmd="$1"
    run_sharded_config "${cmd}"
}
