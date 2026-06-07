#!/usr/bin/env bash

MODEL="haldane-hubbard"
LATTICE=(2 2)
X_PARAM="t2"
Y_PARAM="U"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATTICE_TAG=$(IFS=x; echo "${LATTICE[*]}")
LOG="$HERE/logs/${MODEL}/${LATTICE_TAG}/run-analytic+vqe+iqpe-3d-${X_PARAM}-vs-${Y_PARAM}.json"

PHI=$(python3 -c "import math; print(math.pi/4)")

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    quaph plot "$LOG"
else
    quaph run \
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
        --log-dir "$HERE/logs" \
        --plot-dir "$HERE/plots"
fi
