#!/usr/bin/env bash

MODEL="ssh"
LATTICE=(2)
X_PARAM="n_occ"
Y_PARAM="t2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATTICE_TAG=$(IFS=x; echo "${LATTICE[*]}")
LOG="$HERE/logs/${MODEL}/${LATTICE_TAG}/simulated-ideal-3d-${X_PARAM}-vs-${Y_PARAM}.json"

if [ -f "$LOG" ]; then
    echo "Plotting from existing log..."
    quaph plot "$LOG"
    exit 0
fi

if ! printf 'list\nexit\n' | quaph 2>/dev/null | grep -qE "^[[:space:]]+${MODEL}([[:space:]]|$)"; then
    quaph <<'QUAPH'
register
ssh
SSH
1
1
Lcells
2
A,B
t1
t_1
t2
t_2

hopping
A
B
0

-t1
y
hopping
B
A
1

-t2
y
done
n
y
SPSA
maxiter
@max_iters

y
exit
QUAPH
fi

quaph run simulated-ideal \
    --model "$MODEL" \
    --lattice "${LATTICE[@]}" \
    --x-param "$X_PARAM" \
    --y-param "$Y_PARAM" \
    --y-range 0.0 2.0 0.5 \
    --t1 1.0 \
    --vqe-iters 200 \
    --vqe-layers 2 \
    --vqe-reps 1 \
    --iqpe-time 0.3 \
    --iqpe-trot 2 \
    --iqpe-iters 2 \
    --iqpe-reps 1 \
    --log-dir "$HERE/logs" \
    --plot-dir "$HERE/plots"
