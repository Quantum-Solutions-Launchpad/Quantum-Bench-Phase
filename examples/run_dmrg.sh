#!/usr/bin/env bash

# DMRG vs. exact ground-state energy across a Haldane-model phase diagram.
#
# DMRG requires a working Julia + ITensorMPS toolchain (bundled under
# qbp/julia-dmrg/), so this only runs end-to-end where Julia is available.
# Selecting `analytic` alongside `dmrg` overlays the exact energy as reference.

MODEL="haldane-honeycomb"
LATTICE=(2 2)
X_PARAM="n_occ"
Y_PARAM="t2"
Y_RANGE=(0.0 1.0 0.1)
DMRG_NSWEEPS=4
DMRG_MAXDIMS="20,50,100,200"
DMRG_CUTOFF="1e-9"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/dmrg-n_occ-vs-t2.json"
PLOT="$HERE/plots/${MODEL}/dmrg-n_occ-vs-t2.pdf"

PHI=$(python3 -c "import math; print(math.pi/4)")

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    qbp plot "$LOG"
else
    qbp run \
        --model "$MODEL" \
        --method analytic dmrg \
        --lattice "${LATTICE[@]}" \
        --x-param "$X_PARAM" \
        --y-param "$Y_PARAM" \
        --y-range "${Y_RANGE[@]}" \
        --t1 1.0 --phi "$PHI" --M 0.0 \
        --dmrg-nsweeps "$DMRG_NSWEEPS" \
        --dmrg-maxdims "$DMRG_MAXDIMS" \
        --dmrg-cutoff "$DMRG_CUTOFF" \
        --log-path "$LOG" \
        --plot-path "$PLOT"
fi
