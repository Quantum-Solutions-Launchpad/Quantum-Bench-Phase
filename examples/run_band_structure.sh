#!/usr/bin/env bash

MODEL="haldane"
X_PARAM="kx"
Y_PARAM="ky"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PI=$(python3 -c "import math; print(math.pi)")
STEP=$(python3 -c "import math; print(math.pi/50)")
NEG_PI=$(python3 -c "import math; print(-math.pi)")
PHI=$(python3 -c "import math; print(math.pi/2)")

quaph run \
    --model "$MODEL" \
    --method analytic \
    --x-param "$X_PARAM" \
    --x-range "$NEG_PI" "$PI" "$STEP" \
    --y-param "$Y_PARAM" \
    --y-range "$NEG_PI" "$PI" "$STEP" \
    --t1 1.0 --t2 0.05 --M 0.2 --phi "$PHI" \
    --log-path "$HERE/logs/$MODEL/band-structure-heatmap.json" \
    --plot-path "$HERE/plots/$MODEL/band-structure-heatmap.pdf" \
    --heatmap

quaph run \
    --model "$MODEL" \
    --method analytic \
    --x-param "$X_PARAM" \
    --x-range "$NEG_PI" "$PI" "$STEP" \
    --y-param "$Y_PARAM" \
    --y-range "$NEG_PI" "$PI" "$STEP" \
    --t1 1.0 --t2 0.05 --M 0.2 --phi "$PHI" \
    --log-path "$HERE/logs/$MODEL/band-structure-3d.json" \
    --plot-path "$HERE/plots/$MODEL/band-structure-3d.pdf"
