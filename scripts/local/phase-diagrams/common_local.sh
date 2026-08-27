#!/bin/bash
set -euo pipefail

setup_local_env() {
    REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
    cd "${REPO_ROOT}"

    if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/venv/bin/activate"
    fi

    QBP_CLI="${QBP_CLI:-qbp}"
    if ! command -v "${QBP_CLI}" >/dev/null 2>&1; then
        echo "ERROR: '${QBP_CLI}' not found. Activate the venv or set QBP_CLI." >&2
        exit 1
    fi
    export QBP_CLI

    if [[ "$(uname -s)" == "Darwin" ]]; then
        NCPU="${NCPU:-$(sysctl -n hw.ncpu)}"
    else
        NCPU="${NCPU:-$(nproc)}"
    fi
    export NCPU

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export LOKY_DISABLE_RESOURCE_TRACKER=1
    export JOBLIB_START_METHOD=spawn
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/${USER}-mpl}"
    export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-${NCPU}}"

    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/manuscript-plots}"
    LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs/local}"
    PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots/local}"
    mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${PLOT_DIR}" "${MPLCONFIGDIR}"
}

setup_local_dmrg_env() {
    export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

    JULIA_BIN="${JULIA_BIN:-julia}"
    if ! command -v "${JULIA_BIN}" >/dev/null 2>&1; then
        echo "ERROR: '${JULIA_BIN}' not found on PATH (needed by --method dmrg)." >&2
        exit 1
    fi
    export JULIA_BIN

    if [[ -z "${QBP_JULIA_SYSIMAGE:-}" && ! -f "${REPO_ROOT}/qbp/julia-dmrg/dmrg_sysimage.so" ]]; then
        echo "NOTE: no dmrg_sysimage.so; every cell pays ~45 s of Julia JIT." >&2
        echo "      Build it once: qbp/julia-dmrg/sysimage/build_sysimage.sh" >&2
    fi
}

declare -a STAGED_HAMLIB_FILES=()
stage_hydrogen_hamlib() {
    local work_dir="$1"; shift
    local geometry="$1"; shift
    local basis="$1"; shift
    local -a h_counts=()
    local -a radii=()
    local mode="h"
    local tok
    for tok in "$@"; do
        if [[ "${tok}" == "--" ]]; then mode="r"; continue; fi
        if [[ "${mode}" == "h" ]]; then h_counts+=("${tok}"); else radii+=("${tok}"); fi
    done

    local root="${HAMLIB_ROOT:-${HOME}/hamlib/chemistry/electronic/hydrogen_data}"
    mkdir -p "${work_dir}"
    STAGED_HAMLIB_FILES=()

    local h r src staged nqubits cand
    for h in "${h_counts[@]}"; do
        for r in "${radii[@]}"; do
            src=""
            for cand in \
                "${root}/H${h}_${geometry}/ES_H${h}_${geometry}_R${r}_${basis}_ham.zip" \
                "${root}/H${h}_${geometry}/ES_H${h}_${geometry}_R${r}_${basis}.zip" \
                "${root}/H${h}_${geometry}/ES_H${h}_${geometry}_R${r}_${basis}_ham.hdf5" \
                "${root}/H${h}_${geometry}/ES_H${h}_${geometry}_R${r}_${basis}.hdf5"
            do
                if [[ -f "${cand}" ]]; then src="${cand}"; break; fi
            done
            if [[ -z "${src}" ]]; then
                echo "ERROR: no HamLib file for H${h} ${geometry} R=${r} under ${root}" >&2
                exit 2
            fi

            nqubits=$((2 * h))
            staged="${work_dir}/$(basename "${src%.*}" | sed "s/^ES_H${h}_/ES_nqubits${nqubits}_/").hdf5"

            if [[ ! -f "${staged}" ]]; then
                if [[ "${src}" == *.zip ]]; then
                    python - "${src}" "${staged}" <<'PY'
import shutil, sys, zipfile
src, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    names = [n for n in z.namelist() if not n.endswith("/")]
    h5 = [n for n in names if n.lower().endswith((".h5", ".hdf5", ".he5"))] or names
    with z.open(h5[0]) as fsrc, open(dest, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
PY
                else
                    cp "${src}" "${staged}"
                fi
            fi
            STAGED_HAMLIB_FILES+=("${staged}")
        done
    done
}
