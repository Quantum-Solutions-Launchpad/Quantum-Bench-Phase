#!/usr/bin/env bash

MODEL="haldane"
N_SITES=18
X_PARAM="n_occ"
Y_PARAM="t2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PHI=$(python3 -c "import math; print(math.pi/4)")

quaph run analytic \
    --model "$MODEL" \
    --n-sites "$N_SITES" \
    --x-param "$X_PARAM" \
    --y-param "$Y_PARAM" \
    --y-range 0.0 1.0 0.1 \
    --t1 1.0 --phi "$PHI" --M 0.0 \
    --log-dir "$HERE/logs" \
    --plot-dir "$HERE/plots"
