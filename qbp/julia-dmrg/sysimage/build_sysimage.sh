#!/bin/bash
# Build a PackageCompiler sysimage for the DMRG CLI. Takes 10-30 minutes.
# Output: qbp/julia-dmrg/dmrg_sysimage.so (picked up automatically by _dmrg.py)
#
# Usage: bash qbp/julia-dmrg/sysimage/build_sysimage.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${HERE}/.."
SYSIMAGE="${PROJECT}/dmrg_sysimage.so"

JULIA="${JULIA:-/global/common/software/nersc9/julia/1.11.7/bin/julia}"
[[ -x "${JULIA}" ]] || JULIA="$(command -v julia)"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/pscratch/sd/m/mbao202/julia_depot}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"

echo "julia:     ${JULIA}"
echo "project:   ${PROJECT}"
echo "depot:     ${JULIA_DEPOT_PATH}"
echo "sysimage:  ${SYSIMAGE}"

# PackageCompiler lives in the default (stacked) environment, not the project,
# so the project's Project.toml/Manifest.toml stay untouched.
"${JULIA}" -e 'using Pkg; Pkg.add("PackageCompiler")'

"${JULIA}" --project="${PROJECT}" -e "
using Pkg
Pkg.instantiate()
using PackageCompiler
create_sysimage(
    [\"ITensors\", \"ITensorMPS\", \"JSON\"];
    sysimage_path=\"${SYSIMAGE}\",
    precompile_execution_file=\"${HERE}/precompile_workload.jl\",
)
"

echo "Sysimage built: ${SYSIMAGE}"
echo "Sanity check (should print energy almost instantly):"
tmp_out="$(mktemp -u)/check.json"
time "${JULIA}" --project="${PROJECT}" --sysimage="${SYSIMAGE}" \
    "${PROJECT}/dmrg_itensor_cli.jl" \
    --hamiltonian "${HERE}/sample_hamiltonian_spinless.json" \
    --output "${tmp_out}" \
    --nsweeps 2 --maxdims 10,20 --seed 1 \
    --conserve-qns true --conserve-sz true --initial-state packed
