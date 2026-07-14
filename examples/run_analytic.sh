#!/usr/bin/env bash

MODEL="haldane"
LATTICE=(3 3)
X_PARAM="n_occ"
Y_PARAM="t2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PHI=$(python3 -c "import math; print(math.pi/4)")

qbp run \
    --model "$MODEL" \
    --method analytic \
    --lattice "${LATTICE[@]}" \
    --x-param "$X_PARAM" \
    --y-param "$Y_PARAM" \
    --y-range 0.0 1.0 0.1 \
    --t1 1.0 --phi "$PHI" --M 0.0 \
    --log-path "$HERE/logs/$MODEL/analytic-n_occ-vs-t2.json" \
    --plot-path "$HERE/plots/$MODEL/analytic-n_occ-vs-t2.pdf"
