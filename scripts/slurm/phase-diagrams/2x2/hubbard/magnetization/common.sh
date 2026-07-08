#!/bin/bash
# Shared driver for Hubbard 2x2 magnetization phase diagrams.
#
# Wrappers choose:
#   1. The observable via the SLURM array/TASK_ID: M_stag or M_total
#   2. The methods via METHODS, usually analytic/vqe/dmrg subsets
# Optional: SWEEP_TAG overrides the filename tag (default "n_occ-vs-U").

set -euo pipefail

export PYTHONWARNINGS="ignore:Cannot register"

OBSERVABLES=(M_stag M_total)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
if (( TASK_ID < 0 || TASK_ID >= ${#OBSERVABLES[@]} )); then
    echo "TASK_ID must be 0 (M_stag) or 1 (M_total)." >&2
    exit 1
fi
OBSERVABLE="${OBSERVABLE:-${OBSERVABLES[$TASK_ID]}}"
SWEEP_TAG="${SWEEP_TAG:-n_occ-vs-U}"

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

# Enable intra-shard parallelism for expensive methods (VQE, DMRG).
export QBP_JOBS_PER_SHARD="${QBP_JOBS_PER_SHARD:-2}"

METHODS="${METHODS:-analytic vqe dmrg}"
read -r -a method_array <<< "${METHODS}"
HEATMAP="${HEATMAP:-1}"

# Optional OUT_SUBDIR nests all outputs under hubbard/2x2/<OUT_SUBDIR>/
OUT_LOG_DIR="${LOG_DIR}/hubbard/2x2${OUT_SUBDIR:+/${OUT_SUBDIR}}"
OUT_PLOT_DIR="${PLOT_DIR}/hubbard/2x2${OUT_SUBDIR:+/${OUT_SUBDIR}}"

print_visualizer_header "Running Hubbard 2x2 magnetization phase diagram (${OBSERVABLE}, ${SWEEP_TAG}, ${PIPELINE})"
echo "  sweep:       n_occ [${HUBBARD_N_OCC_START:-0} ${HUBBARD_N_OCC_END:-16} ${HUBBARD_N_OCC_STEP:-1}] vs U [${HUBBARD_U_START:-0.0} ${HUBBARD_U_END:-10.0} ${HUBBARD_U_STEP:-0.5}]"
echo "  pipeline:    ${PIPELINE}${backend_args[1]:+ (backend: ${backend_args[1]})}"
echo "  methods:     ${METHODS} (running in parallel)"
echo "  observable:  ${OBSERVABLE}"
echo "  plot format: $([[ "${HEATMAP}" == "1" ]] && echo heatmap || echo 3d)"
echo "  fixed:       t=${HUBBARD_T:-1.0}"
echo "  iqpe:        disabled (operator observables are unsupported)"
echo "  dmrg QNs:    particle number conserved; Sz allowed to optimize"
echo "  dmrg seed:   ${DMRG_INITIAL_STATE:-neel} product state"

append_hubbard_magnetization_cmd_args() {
    local array_name="$1"
    local -n cmd_ref="${array_name}"

    cmd_ref+=(
        --model hubbard
        --lattice 2 2
        --observable "${OBSERVABLE}"
        --x-param n_occ
        --x-range "${HUBBARD_N_OCC_START:-0}" "${HUBBARD_N_OCC_END:-16}" "${HUBBARD_N_OCC_STEP:-1}"
        --y-param U
        --y-range "${HUBBARD_U_START:-0.0}" "${HUBBARD_U_END:-10.0}" "${HUBBARD_U_STEP:-0.5}"
        --t "${HUBBARD_T:-1.0}"
    )
    if (( ${#backend_args[@]} )); then
        cmd_ref+=("${backend_args[@]}")
    fi
    append_optional_vqe_iqpe_args "${array_name}"
    append_dmrg_args "${array_name}"
    cmd_ref+=(
        --no-dmrg-conserve-sz
        --dmrg-initial-state "${DMRG_INITIAL_STATE:-neel}"
    )
}

run_method_parallel() {
    local method="$1"
    local method_status=0

    cmd=(
        --method "$method"
    )
    append_hubbard_magnetization_cmd_args cmd

    if [[ "${HEATMAP}" == "1" ]]; then
        cmd+=(--heatmap)
    fi

    LOG_PATH="${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-${OBSERVABLE}-${SWEEP_TAG}.json"
    PLOT_PATH="${OUT_PLOT_DIR}/simulated-${PIPELINE}-${method}-${OBSERVABLE}-${SWEEP_TAG}.pdf"
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
    COMBINED_LOG="${OUT_LOG_DIR}/simulated-${PIPELINE}-all-methods-${OBSERVABLE}-${SWEEP_TAG}.json"
    COMBINED_PLOT="${OUT_PLOT_DIR}/simulated-${PIPELINE}-all-methods-${OBSERVABLE}-${SWEEP_TAG}.pdf"
    COMBINED_PROGRESS="${COMBINED_LOG%.json}.progress.jsonl"

    : > "${COMBINED_PROGRESS}.tmp"
    for method in "${method_array[@]}"; do
        cat "${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-${OBSERVABLE}-${SWEEP_TAG}.progress.jsonl" \
            >> "${COMBINED_PROGRESS}.tmp"
    done
    mv "${COMBINED_PROGRESS}.tmp" "${COMBINED_PROGRESS}"

    cmd=(
        --method "${method_array[@]}"
    )
    append_hubbard_magnetization_cmd_args cmd
    append_qbp_output_paths cmd "${COMBINED_LOG}" "${COMBINED_PLOT}"

    echo "Building combined all-methods plot"
    ${QBP_CLI} run "${cmd[@]}" --aggregate-only
fi

# ------------------------------------------------------------- verify outputs
echo ""
echo "=== Output verification ==="
fail=0
for method in "${method_array[@]}"; do
    summary="${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-${OBSERVABLE}-${SWEEP_TAG}.json"
    plot="${OUT_PLOT_DIR}/simulated-${PIPELINE}-${method}-${OBSERVABLE}-${SWEEP_TAG}.pdf"
    progress="${OUT_LOG_DIR}/simulated-${PIPELINE}-${method}-${OBSERVABLE}-${SWEEP_TAG}.progress.jsonl"
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
    echo "FINAL PLOT: ${OUT_PLOT_DIR}/simulated-${PIPELINE}-${method_array[0]}-${OBSERVABLE}-${SWEEP_TAG}.pdf"
fi
echo "Per-method outputs: ${OUT_LOG_DIR}/simulated-${PIPELINE}-<method>-${OBSERVABLE}-${SWEEP_TAG}.{json,pdf,progress.jsonl}"
echo "=================================================================="
