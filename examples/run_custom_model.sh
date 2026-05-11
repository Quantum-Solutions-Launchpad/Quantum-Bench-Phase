#!/usr/bin/env bash

MODEL="ssh"
N_SITES=4
X_PARAM="n_occ"
Y_PARAM="t2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/logs/${MODEL}/${N_SITES}-sites/simulated-ideal-${X_PARAM}-vs-${Y_PARAM}.json"

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
t1
1.0

t1
t_1
t2
t_2

import numpy as np
def hamiltonian_matrix(n_sites, t1, t2):
    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)
    for i in range(n_sites - 1):
        t = t1 if i % 2 == 0 else t2
        for s in range(spin):
            s1 = i * spin + s
            s2 = (i + 1) * spin + s
            H[s1, s2] -= t
            H[s2, s1] -= t
    return H
END
from qiskit_nature.second_q.operators import FermionicOp
def fermionic_hamiltonian(n_sites, *, t1, t2):
    spin = 2
    hamiltonian = 0.0 * FermionicOp({})
    for i in range(n_sites - 1):
        t = t1 if i % 2 == 0 else t2
        for s in range(spin):
            s1 = i * spin + s
            s2 = (i + 1) * spin + s
            hamiltonian -= FermionicOp({
                f"+_{s1} -_{s2}": t,
                f"+_{s2} -_{s1}": t,
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
t2
0.0
2.0
0.25
y
exit
QUAPH
fi

quaph run simulated-ideal \
    --model "$MODEL" \
    --n-sites "$N_SITES" \
    --x-param "$X_PARAM" \
    --y-param "$Y_PARAM" \
    --vqe-iters 500 \
    --vqe-layers 2 \
    --vqe-reps 1 \
    --iqpe-time 0.3 \
    --iqpe-trot 2 \
    --iqpe-iters 4 \
    --iqpe-reps 1 \
    --log-dir "$HERE/logs" \
    --plot-dir "$HERE/plots"
