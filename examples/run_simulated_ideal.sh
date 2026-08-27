#!/usr/bin/env bash

MODEL="haldane-hubbard-honeycomb"
LATTICE=(2 2)
X_PARAM="t2"
Y_PARAM="U"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/sim-ideal-t2-vs-U.json"
PLOT="$HERE/plots/${MODEL}/sim-ideal-t2-vs-U.pdf"

PHI=$(python3 -c "import math; print(math.pi/4)")

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    qbp plot "$LOG"
else
    qbp run \
        --model "$MODEL" \
        --method analytic vqe iqpe \
        --lattice "${LATTICE[@]}" \
        --x-param "$X_PARAM" \
        --x-range 0.0 1.5 0.3 \
        --y-param "$Y_PARAM" \
        --y-range 0.0 4.0 1.0 \
        --t1 1.0 --phi "$PHI" --M 0.0 \
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
