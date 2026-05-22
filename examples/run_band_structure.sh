#!/usr/bin/env bash

MODEL="haldane"
X_PARAM="kx"
Y_PARAM="ky"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PI=$(python3 -c "import math; print(math.pi)")
STEP=$(python3 -c "import math; print(math.pi/50)")
NEG_PI=$(python3 -c "import math; print(-math.pi)")
PHI=$(python3 -c "import math; print(math.pi/2)")

quaph run analytic \
    --model "$MODEL" \
    --x-param "$X_PARAM" \
    --x-range "$NEG_PI" "$PI" "$STEP" \
    --y-param "$Y_PARAM" \
    --y-range "$NEG_PI" "$PI" "$STEP" \
    --t1 1.0 --t2 0.05 --M 0.2 --phi "$PHI" \
    --log-dir "$HERE/logs" \
    --plot-dir "$HERE/plots" \
    --heatmap

quaph run analytic \
    --model "$MODEL" \
    --x-param "$X_PARAM" \
    --x-range "$NEG_PI" "$PI" "$STEP" \
    --y-param "$Y_PARAM" \
    --y-range "$NEG_PI" "$PI" "$STEP" \
    --t1 1.0 --t2 0.05 --M 0.2 --phi "$PHI" \
    --log-dir "$HERE/logs" \
    --plot-dir "$HERE/plots"
