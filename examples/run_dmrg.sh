#!/usr/bin/env bash

MODEL="haldane"
LATTICE=(2 2)
X_PARAM="n_occ"
Y_PARAM="t2"
Y_RANGE=(0.0 1.0 0.1)
DMRG_NSWEEPS=4
DMRG_MAXDIMS="20,50,100,200"
DMRG_CUTOFF="1e-9"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATTICE_TAG=$(IFS=x; echo "${LATTICE[*]}")
LOG_DIR="$HERE/logs"
PLOT_DIR="$HERE/plots"
DMRG_LOG="$LOG_DIR/${MODEL}/${LATTICE_TAG}/dmrg/dmrg-${X_PARAM}-vs-${Y_PARAM}.json"
COMPARE_LOG="$LOG_DIR/${MODEL}/${LATTICE_TAG}/compare-${X_PARAM}-vs-${Y_PARAM}.json"

PHI=$(python3 -c "import math; print(math.pi/4)")

if [ -f "$DMRG_LOG" ]; then
    echo "DMRG log already exists: $DMRG_LOG"
else
    quaph run dmrg \
        --model "$MODEL" \
        --lattice "${LATTICE[@]}" \
        --x-param "$X_PARAM" \
        --y-param "$Y_PARAM" \
        --y-range "${Y_RANGE[@]}" \
        --t1 1.0 --phi "$PHI" --M 0.0 \
        --nsweeps "$DMRG_NSWEEPS" \
        --maxdims "$DMRG_MAXDIMS" \
        --cutoff "$DMRG_CUTOFF" \
        --log-dir "$LOG_DIR" \
        --plot-dir "$PLOT_DIR"
fi

if [ -f "$COMPARE_LOG" ]; then
    echo "Plotting from existing compare log..."
    quaph plot "$COMPARE_LOG"
else
    quaph run compare \
        --model "$MODEL" \
        --lattice "${LATTICE[@]}" \
        --x-param "$X_PARAM" \
        --y-param "$Y_PARAM" \
        --y-range "${Y_RANGE[@]}" \
        --t1 1.0 --phi "$PHI" --M 0.0 \
        --algorithms exact dmrg \
        --quantum-pipeline ideal \
        --dmrg-nsweeps "$DMRG_NSWEEPS" \
        --dmrg-maxdims "$DMRG_MAXDIMS" \
        --dmrg-cutoff "$DMRG_CUTOFF" \
        --log-dir "$LOG_DIR" \
        --plot-dir "$PLOT_DIR"
fi

# To include quantum algorithms in the comparison, add vqe and/or iqpe to
# --algorithms and provide the matching VQE/IQPE flags:
#   --algorithms exact vqe iqpe dmrg \
#   --vqe-iters 200 --vqe-layers 2 --vqe-reps 1 \
#   --iqpe-time 0.2 --iqpe-trot 2 --iqpe-iters 2 --iqpe-reps 1
