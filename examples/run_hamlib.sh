#!/usr/bin/env bash

# Transverse-Field Ising Model (TFIM) from the Hamlib dataset.
#
# Plots the analytic ground-state energy sweeping both system size (Lx) and
# transverse field (h) for a 1D open-boundary chain.  The quantum phase
# transition at h = 1 is clearly visible in the 2D line plot.

HAMLIB_PATH="/Users/adamgodel/hamlib/condensedmatter/tfim/tfim.zip"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 3D surface: ground-state energy vs Lx and h
qbp run \
    --method analytic \
    --qubit-operator "$HAMLIB_PATH" \
    --x-param Lx --x-range 4 12 \
    --y-param h \
    --select 1D nonpbc \
    --log-path "$HERE/logs/tfim/Lx-vs-h.json" \
    --plot-path "$HERE/plots/tfim/Lx-vs-h.pdf"

# Heatmap: same data, alternative view
qbp run \
    --method analytic \
    --qubit-operator "$HAMLIB_PATH" \
    --x-param Lx --x-range 4 12 \
    --y-param h \
    --select 1D nonpbc \
    --heatmap \
    --log-path "$HERE/logs/tfim/Lx-vs-h-heatmap.json" \
    --plot-path "$HERE/plots/tfim/Lx-vs-h-heatmap.pdf"

# 2D line: energy vs transverse field at fixed Lx=8
qbp run \
    --method analytic \
    --qubit-operator "$HAMLIB_PATH" \
    --x-param h \
    --select 1D nonpbc Lx-8 \
    --log-path "$HERE/logs/tfim/h.json" \
    --plot-path "$HERE/plots/tfim/h.pdf"
