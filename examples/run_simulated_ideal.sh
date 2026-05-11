#!/usr/bin/env bash

MODEL="haldane-hubbard"
N_SITES=6
X_PARAM="t2"
Y_PARAM="U"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/${N_SITES}-sites/simulated-ideal-${X_PARAM}-vs-${Y_PARAM}.json"

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    quaph plot "$LOG"
else
    quaph run simulated-ideal \
        --model "$MODEL" \
        --n-sites "$N_SITES" \
        --x-param "$X_PARAM" \
        --y-param "$Y_PARAM" \
        --vqe-iters 10000 \
        --vqe-layers 5 \
        --vqe-reps 10 \
        --iqpe-time 0.2 \
        --iqpe-trot 5 \
        --iqpe-iters 8 \
        --iqpe-reps 20 \
        --log-dir "$HERE/logs" \
        --plot-dir "$HERE/plots"
fi
