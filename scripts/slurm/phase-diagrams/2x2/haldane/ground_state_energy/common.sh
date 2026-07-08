#!/bin/bash
# Shared driver for Haldane 2x2 ground-state-energy phase diagrams.
#
# Wrappers only choose:
#   1. The sweep axes:  X_PARAM/X_START/X_END/X_STEP, Y_PARAM/Y_START/Y_END/Y_STEP
#      (params: M, phi, t1, t2, n_occ; non-swept params get fixed defaults below)
#   2. The pipeline:    PIPELINE=ideal (default) or noisy
#      (noisy uses NOISY_BACKEND, default FakeSherbrooke)
# Optional: SWEEP_TAG overrides the filename tag (default "<X>-vs-<Y>").

set -euo pipefail

export PYTHONWARNINGS="ignore:Cannot register"

: "${X_PARAM:?wrapper must set X_PARAM}"
: "${X_START:?wrapper must set X_START}"
: "${X_END:?wrapper must set X_END}"
: "${X_STEP:?wrapper must set X_STEP}"
: "${Y_PARAM:?wrapper must set Y_PARAM}"
: "${Y_START:?wrapper must set Y_START}"
: "${Y_END:?wrapper must set Y_END}"
: "${Y_STEP:?wrapper must set Y_STEP}"

SWEEP_TAG="${SWEEP_TAG:-${X_PARAM}-vs-${Y_PARAM}}"

PIPELINE="${PIPELINE:-ideal}"
backend_args=()
if [[ "${PIPELINE}" == "noisy" ]]; then
    backend_args=(--backend "${NOISY_BACKEND:-FakeSherbrooke}")
elif [[ "${PIPELINE}" != "ideal" ]]; then
    echo "PIPELINE must be 'ideal' or 'noisy' (got '${PIPELINE}')" >&2
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
source "${REPO_ROOT}/scripts/slurm/phase-diagrams/common_visualizer.sh"
setup_visualizer_env
setup_visualizer_dmrg_env "${SLURM_CPUS_PER_TASK:-8}"

# Enable intra-shard parallelism for expensive methods (VQE, IQPE, DMRG)
export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

# Fixed model parameters for whichever axes are not being swept
declare -A FIXED=(
    [t1]="${HALDANE_T1:-1.0}"
    [t2]="${HALDANE_T2:-1.0}"
    [M]="${HALDANE_M:-0.0}"
    [phi]="${HALDANE_PHI:-0.785398369}"
)
fixed_args=()
fixed_desc=""
for p in t1 t2 M phi; do
    if [[ "${p}" != "${X_PARAM}" && "${p}" != "${Y_PARAM}" ]]; then
        fixed_args+=(--"${p}" "${FIXED[$p]}")
        fixed_desc+="${p}=${FIXED[$p]} "
    fi
done

# Optional OUT_SUBDIR nests all outputs under haldane/2x2/<OUT_SUBDIR>/
OUT_LOG_DIR="${LOG_DIR}/haldane/2x2${OUT_SUBDIR:+/${OUT_SUBDIR}}"
OUT_PLOT_DIR="${PLOT_DIR}/haldane/2x2${OUT_SUBDIR:+/${OUT_SUBDIR}}"

METHODS="${METHODS:-iqpe vqe dmrg analytic}"  # slowest to fastest
read -r -a method_array <<< "${METHODS}"
HEATMAP="${HEATMAP:-0}"

print_visualizer_header "Running Haldane 2x2 ground-state energy phase diagram (${SWEEP_TAG}, ${PIPELINE})"
echo "  sweep:       ${X_PARAM} [${X_START} ${X_END} ${X_STEP}] vs ${Y_PARAM} [${Y_START} ${Y_END} ${Y_STEP}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  methods:     ${METHODS} (running in parallel)"
echo "  plot format: $([[ "${HEATMAP}" == "1" ]] && echo heatmap || echo 3d)"
echo "  fixed:       ${fixed_desc}"

run_method_parallel() {
    local method="$1"
    local method_status=0

    cmd=(
        --method "$method"
        --observable E
        --model haldane
        --lattice 2 2
        --x-param "${X_PARAM}"
        --x-range "${X_START}" "${X_END}" "${X_STEP}"
        --y-param "${Y_PARAM}"
        --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
        "${fixed_args[@]}"
    )
    if (( ${#backend_args[@]} )); then
        cmd+=("${backend_args[@]}")
    fi
    append_vqe_iqpe_args cmd
    append_dmrg_args cmd

    if [[ "${HEATMAP}" == "1" ]]; then
        cmd+=(--heatmap)
    fi

    LOG_PATH="${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-E-${SWEEP_TAG}.json"
    PLOT_PATH="${OUT_PLOT_DIR}/simulated-${PIPELINE}-${method}-E-${SWEEP_TAG}.pdf"
    append_qbp_output_paths cmd "${LOG_PATH}" "${PLOT_PATH}"

    echo "Starting method: $method"
    run_visualizer_sharded_cmd cmd
    method_status=$?

    if (( method_status == 0 )); then
        echo "Completed ${method} run at $(date)"
        echo "  summary: ${LOG_PATH}"
        echo "  plot:    ${PLOT_PATH}"
    else
        echo "ERROR: ${method} run failed with status ${method_status}" >&2
        return "${method_status}"
    fi
}

pids=()
for method in "${method_array[@]}"; do
    run_method_parallel "$method" &
    pids+=("$!")
done

exit_status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        exit_status=1
    fi
done

if (( exit_status != 0 )); then
    echo "One or more methods failed" >&2
    exit 1
fi

# ------------------------------------------- combined single plot (all methods)
# Heatmaps in QBP are intentionally single-method, so only build a combined
# figure for 3D runs.
if (( ${#method_array[@]} > 1 )) && [[ "${HEATMAP}" != "1" ]]; then
    COMBINED_LOG="${OUT_LOG_DIR}/simulated-${PIPELINE}-all-methods-E-${SWEEP_TAG}.json"
    COMBINED_PLOT="${OUT_PLOT_DIR}/simulated-${PIPELINE}-all-methods-E-${SWEEP_TAG}.pdf"
    COMBINED_PROGRESS="${COMBINED_LOG%.json}.progress.jsonl"

    : > "${COMBINED_PROGRESS}.tmp"
    for method in "${method_array[@]}"; do
        cat "${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-E-${SWEEP_TAG}.progress.jsonl" \
            >> "${COMBINED_PROGRESS}.tmp"
    done
    mv "${COMBINED_PROGRESS}.tmp" "${COMBINED_PROGRESS}"

    cmd=(
        --method "${method_array[@]}"
        --observable E
        --model haldane
        --lattice 2 2
        --x-param "${X_PARAM}"
        --x-range "${X_START}" "${X_END}" "${X_STEP}"
        --y-param "${Y_PARAM}"
        --y-range "${Y_START}" "${Y_END}" "${Y_STEP}"
        "${fixed_args[@]}"
    )
    if (( ${#backend_args[@]} )); then
        cmd+=("${backend_args[@]}")
    fi
    append_vqe_iqpe_args cmd
    append_dmrg_args cmd
    append_qbp_output_paths cmd "${COMBINED_LOG}" "${COMBINED_PLOT}"

    echo "Building combined all-methods plot"
    ${QBP_CLI} run "${cmd[@]}" --aggregate-only
fi

# ------------------------------------------------------------- verify outputs
echo ""
echo "=== Output verification ==="
fail=0
for method in "${method_array[@]}"; do
    summary="${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-E-${SWEEP_TAG}.json"
    plot="${OUT_PLOT_DIR}/simulated-${PIPELINE}-${method}-E-${SWEEP_TAG}.pdf"
    progress="${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-E-${SWEEP_TAG}.progress.jsonl"
    [[ -s "${summary}" ]] && echo "OK  summary:  ${summary}" || { echo "MISSING summary:  ${summary}"; fail=1; }
    [[ -s "${plot}" ]] && echo "OK  plot:     ${plot}" || { echo "MISSING plot:     ${plot}"; fail=1; }
    if [[ -s "${progress}" ]]; then
        echo "OK  progress: ${progress} ($(wc -l < "${progress}") cells logged)"
    else
        echo "MISSING progress: ${progress}"; fail=1
    fi
done
if (( ${#method_array[@]} > 1 )) && [[ "${HEATMAP}" != "1" ]]; then
    [[ -s "${COMBINED_PLOT}" ]] && echo "OK  combined plot: ${COMBINED_PLOT}" || { echo "MISSING combined plot: ${COMBINED_PLOT}"; fail=1; }
fi
if (( fail )); then
    echo "OUTPUT VERIFICATION FAILED: some outputs missing" >&2
    exit 1
fi

echo ""
echo "All methods completed at $(date)"
echo "=================================================================="
if (( ${#method_array[@]} > 1 )) && [[ "${HEATMAP}" != "1" ]]; then
    echo "FINAL PLOT (all methods): ${COMBINED_PLOT}"
    echo "  summary:                ${COMBINED_LOG}"
else
    echo "FINAL PLOT: ${OUT_PLOT_DIR}/simulated-${PIPELINE}-${method_array[0]}-E-${SWEEP_TAG}.pdf"
fi
echo "Per-method outputs: ${OUT_LOG_DIR}/simulated-${PIPELINE}-<method>-E-${SWEEP_TAG}.{json,pdf,progress.jsonl}"
echo "=================================================================="
