#!/usr/bin/env bash

MODEL="hubbard"
LATTICE=(2 2)
X_PARAM="n_occ"
Y_PARAM="U"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/sim-noisy-n_occ-vs-U.json"
PLOT="$HERE/plots/${MODEL}/sim-noisy-n_occ-vs-U.pdf"

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    quaph plot "$LOG"
else
    quaph run \
        --model "$MODEL" \
        --method analytic vqe iqpe \
        --noisy \
        --lattice "${LATTICE[@]}" \
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
        --log-path "$LOG" \
        --plot-path "$PLOT"
fi
