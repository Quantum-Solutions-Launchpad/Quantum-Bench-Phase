import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA

from quaph._model import Model


def _get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _build_H_matrix(n_sites, t, U):
    lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)
    for i, j in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            H[s1, s2] -= t
            H[s2, s1] -= t
    return H


def _fermionic_hamiltonian(n_sites, *, t, U):
    lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    spin = 2

    hamiltonian = 0.0 * FermionicOp({})

    for i, j in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            hamiltonian -= FermionicOp({
                f"+_{s1} -_{s2}": t,
                f"+_{s2} -_{s1}": t
            })

    for i in range(n_sites):
        spin_up = i * spin + 0
        spin_down = i * spin + 1
        hamiltonian += FermionicOp({
            f"+_{spin_up} -_{spin_up} +_{spin_down} -_{spin_down}": U
        })

    return hamiltonian


def _mean_field_correction(n_sites, n_occ, **params):
    U = params['U']
    if U == 0:
        return 0.0
    return U * n_sites * (n_occ / (2 * n_sites)) ** 2


model = Model(
    name="hubbard",
    display_name="Hubbard",
    default_params={"t": 1.0},
    param_labels={"t": "t", "U": "U"},
    hamiltonian_matrix=_build_H_matrix,
    fermionic_hamiltonian=_fermionic_hamiltonian,
    get_optimizer=_get_optimizer,
    mean_field_correction=_mean_field_correction,
    sweep_defaults={"y": {"param": "U", "range": (0.0, 4.0, 0.5)}},
)
