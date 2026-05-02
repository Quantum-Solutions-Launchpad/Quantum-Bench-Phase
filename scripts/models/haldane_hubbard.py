import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA


NAME = "haldane-hubbard"
DISPLAY_NAME = "Haldane\u2013Hubbard"
DEFAULT_PARAMS = {"t1": 1.0, "phi": np.pi/4, "M": 0.0}
PARAM_LABELS = {"t1": "t_1", "U": "U", "t2": "t_2", "phi": "\\phi", "M": "M"}
SWEEP_DEFAULTS = {
    "x": {"param": "t2", "range": (0.0, 1.5, 0.1)},
    "y": {"param": "U",  "range": (0.0, 4.0, 0.5)},
}

def get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)

def _build_H_matrix(n_sites, t1, U, t2, phi, M):
    nn_lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    nnn_lattice = [(i, (i + 2) % n_sites) for i in range(n_sites)]
    spin = 2

    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)

    for i in range(n_sites):
        for s in range(spin):
            H[i * spin + s, i * spin + s] += M if i % 2 == 0 else -M

    for i, j in nn_lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            H[s1, s2] -= t1
            H[s2, s1] -= t1

    for i, j in nnn_lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            H[s1, s2] -= t2 * np.exp(1j * phi)
            H[s2, s1] -= t2 * np.exp(-1j * phi)

    return H

def fermionic_hamiltonian(n_sites, *, t1, U, t2, phi, M):
    nn_lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    nnn_lattice = [(i, (i + 2) % n_sites) for i in range(n_sites)]
    spin = 2

    hamiltonian = 0.0 * FermionicOp({})

    for i in range(n_sites):
        for s in range(spin):
            idx = i * spin + s
            hamiltonian += FermionicOp({f"+_{idx} -_{idx}": M if i % 2 == 0 else -M})

    for i, j in nn_lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            hamiltonian -= FermionicOp({
                f"+_{s1} -_{s2}": t1,
                f"+_{s2} -_{s1}": t1
            })

    for i, j in nnn_lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            hamiltonian -= FermionicOp({
                f"+_{s1} -_{s2}": t2 * np.exp(1j * phi),
                f"+_{s2} -_{s1}": t2 * np.exp(-1j * phi)
            })

    for i in range(n_sites):
        spin_up = i * spin + 0
        spin_down = i * spin + 1
        hamiltonian += FermionicOp({
            f"+_{spin_up} -_{spin_up} +_{spin_down} -_{spin_down}": U
        })

    return hamiltonian

def mean_field_correction(n_sites, n_occ, **params):
    U = params['U']
    if U == 0:
        return 0.0
    return U * n_sites * (n_occ / (2 * n_sites)) ** 2
