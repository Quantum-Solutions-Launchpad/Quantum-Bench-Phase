#!/usr/bin/env bash
# Custom models require Python registration and cannot be invoked via the CLI
# directly, so this script delegates to Python for the simulation step.

MODEL="ssh"
N_SITES=4
X_PARAM="n_occ"
Y_PARAM="t2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/${N_SITES}-sites/simulated-ideal-${X_PARAM}-vs-${Y_PARAM}.json"

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    quaph plot "$LOG"
else
    python "$HERE/run_custom_model.py"
fi
