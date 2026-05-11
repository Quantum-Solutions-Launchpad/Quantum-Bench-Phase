import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA

from quaph._model import Model


def _get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _build_H_matrix(n_sites, t1, t2, phi, M):
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)] + [(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)
    for i in range(n_sites):
        for s in range(spin):
            H[i * spin + s, i * spin + s] += M if i % 2 == 0 else -M
    for i, j, order in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            if order == 0:
                H[s1, s2] -= t1
                H[s2, s1] -= t1
            else:
                H[s1, s2] -= t2 * np.exp(1j * phi)
                H[s2, s1] -= t2 * np.exp(-1j * phi)
    return H


def _fermionic_hamiltonian(n_sites, *, t1, t2, phi, M):
    lattice = [(i, (i + 1) % n_sites, 0) for i in range(n_sites)] + [(i, (i + 2) % n_sites, 1) for i in range(n_sites)]
    spin = 2

    hamiltonian = 0.0 * FermionicOp({})
    for i in range(n_sites):
        for s in range(spin):
            idx = i * spin + s
            hamiltonian += FermionicOp({f"+_{idx} -_{idx}": M if i % 2 == 0 else -M})
    for i, j, order in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            if order == 0:
                hamiltonian -= FermionicOp({
                    f"+_{s1} -_{s2}": t1,
                    f"+_{s2} -_{s1}": t1
                })
            else:
                hamiltonian -= FermionicOp({
                    f"+_{s1} -_{s2}": t2 * np.exp(1j * phi),
                    f"+_{s2} -_{s1}": t2 * np.exp(-1j * phi)
                })

    return hamiltonian


model = Model(
    name="haldane",
    display_name="Haldane",
    default_params={"t1": 1.0, "phi": np.pi / 4, "M": 0.0},
    param_labels={"t1": "t_1", "t2": "t_2", "phi": "\\phi", "M": "M"},
    hamiltonian_matrix=_build_H_matrix,
    fermionic_hamiltonian=_fermionic_hamiltonian,
    get_optimizer=_get_optimizer,
    sweep_defaults={"y": {"param": "t2", "range": (0.0, 1.0, 0.1)}},
)
