#!/usr/bin/env bash

MODEL="ssh"
LATTICE=(4)
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
L
1
t1
t_1
t2
t_2

import numpy as np
def hamiltonian_matrix(lattice, t1, t2):
    L, = lattice
    H = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        t = t1 if i % 2 == 0 else t2
        H[i, i + 1] -= t
        H[i + 1, i] -= t
    return H
END
from qiskit_nature.second_q.operators import FermionicOp
def fermionic_hamiltonian(lattice, *, t1, t2):
    L, = lattice
    hamiltonian = 0.0 * FermionicOp({})
    for i in range(L - 1):
        t = t1 if i % 2 == 0 else t2
        hamiltonian -= FermionicOp({
            f"+_{i} -_{i + 1}": t,
            f"+_{i + 1} -_{i}": t,
        })
    return hamiltonian
END
from qiskit_algorithms.optimizers import SPSA
def get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)
END
skip
skip
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
