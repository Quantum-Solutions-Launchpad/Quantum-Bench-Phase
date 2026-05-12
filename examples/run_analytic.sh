#!/usr/bin/env bash

MODEL="haldane"
N_SITES=18
X_PARAM="n_occ"
Y_PARAM="t2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

quaph run analytic \
    --model "$MODEL" \
    --n-sites "$N_SITES" \
    --x-param "$X_PARAM" \
    --y-param "$Y_PARAM" \
    --log-dir "$HERE/logs" \
    --plot-dir "$HERE/plots"
