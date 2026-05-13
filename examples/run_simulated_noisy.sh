#!/usr/bin/env bash

MODEL="hubbard"
N_SITES=8
X_PARAM="n_occ"
Y_PARAM="U"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/${N_SITES}-sites/simulated-noisy-3d-${X_PARAM}-vs-${Y_PARAM}.json"

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    quaph plot "$LOG"
else
    quaph run simulated-noisy \
        --model "$MODEL" \
        --n-sites "$N_SITES" \
        --x-param "$X_PARAM" \
        --y-param "$Y_PARAM" \
        --y-range 0.0 4.0 1.0 \
        --t 1.0 \
        --vqe-iters 200 \
        --vqe-layers 2 \
        --vqe-reps 1 \
        --iqpe-time 0.2 \
        --iqpe-trot 2 \
        --iqpe-iters 2 \
        --iqpe-reps 1 \
        --log-dir "$HERE/logs" \
        --plot-dir "$HERE/plots"
fi
