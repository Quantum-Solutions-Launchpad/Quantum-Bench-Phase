#!/usr/bin/env bash
set -euo pipefail

MODEL="hubbard"
N_SITES=6
X_PARAM="n_occ"
Y_PARAM="U"

LOG="logs/${MODEL}/${N_SITES}-sites/simulated-noisy-${X_PARAM}-vs-${Y_PARAM}.json"

if [ -f "$LOG" ]; then
    quaph plot "$LOG"
else
    quaph run simulated-noisy "$MODEL" \
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
        --log-dir logs \
        --plot-dir plots
fi
