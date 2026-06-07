#!/usr/bin/env bash

# Transverse-Field Ising Model (TFIM) from the Hamlib dataset.
#
# Plots the analytic ground-state energy sweeping both system size (Lx) and
# transverse field (h) for a 1D open-boundary chain.  The quantum phase
# transition at h = 1 is clearly visible in the 2D line plot.

HAMLIB_PATH="/Users/adamgodel/hamlib/condensedmatter/tfim/tfim.zip"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 3D surface: ground-state energy vs Lx and h
quaph run analytic \
    --qubit-operator "$HAMLIB_PATH" \
    --x-param Lx --x-range 4 12 \
    --y-param h \
    --select 1D nonpbc \
    --log-dir "$HERE/logs/tfim" \
    --plot-dir "$HERE/plots/tfim"

# Heatmap: same data, alternative view
quaph run analytic \
    --qubit-operator "$HAMLIB_PATH" \
    --x-param Lx --x-range 4 12 \
    --y-param h \
    --select 1D nonpbc \
    --heatmap \
    --log-dir "$HERE/logs/tfim" \
    --plot-dir "$HERE/plots/tfim"

# 2D line: energy vs transverse field at fixed Lx=8
quaph run analytic \
    --qubit-operator "$HAMLIB_PATH" \
    --x-param h \
    --select 1D nonpbc Lx-8 \
    --log-dir "$HERE/logs/tfim" \
    --plot-dir "$HERE/plots/tfim"
