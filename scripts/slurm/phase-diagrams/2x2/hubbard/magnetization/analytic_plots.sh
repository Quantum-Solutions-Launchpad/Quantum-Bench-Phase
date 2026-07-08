#!/bin/bash
#SBATCH -J hubbard-2x2-analytic-plots
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -A m5027
#SBATCH -o /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/m/mbao202/NNL-P7/scripts/logs/slurm/%x-%j.err

set -euo pipefail

# Re-render the existing analytical Hubbard 2x2 result JSONs. This does not
# repeat the exact many-body diagonalization.

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/m/mbao202/NNL-P7}"
cd "${REPO_ROOT}"

module load python
source "${REPO_ROOT}/venv/bin/activate"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}-mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/examples/logs/hubbard/2x2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/manuscript-plots/plots/hubbard/2x2}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/scripts/logs/slurm"

echo "Rendering Hubbard 2x2 analytical magnetization heatmaps"
echo "  job id:      ${SLURM_JOB_ID:-manual}"
echo "  input dir:   ${INPUT_DIR}"
echo "  output dir:  ${OUTPUT_DIR}"

for observable in M_stag M_total; do
    input="${INPUT_DIR}/analytic-${observable}-heatmap-n_occ-vs-U.json"
    output="${OUTPUT_DIR}/analytic-${observable}-heatmap-n_occ-vs-U.pdf"

    if [[ ! -f "${input}" ]]; then
        echo "Missing analytical result: ${input}" >&2
        exit 1
    fi

    python - "${input}" "${output}" <<'PY'
import json
import sys

import numpy as np

from qbp._run import RunResult

input_path, output_path = sys.argv[1:3]
with open(input_path) as f:
    data = json.load(f)

method = next(iter(data["result"]))
block = data["result"][method]
nx = len(data["x_values"])
ny = len(data["y_values"])
grid = np.array([
    [float(block[str(ix)][str(iy)]) for iy in range(ny)]
    for ix in range(nx)
])

result = RunResult(
    model_name=data["parameters"]["model"],
    lattice=tuple(data["parameters"]["lattice"]),
    x_param=data["x_param"],
    y_param=data["y_param"],
    x_values=data["x_values"],
    y_values=data["y_values"],
    methods=[method],
    grids={method: grid},
    plot_format="heatmap",
    observable=data["observable"],
    backend_label="ideal",
    _model_params=data["parameters"].get("model_params", {}),
)
result.plot(output_path=output_path, hide_plot=True)
PY
    echo "Created ${output}"
done

echo "Completed analytical heatmap rendering at $(date)"
