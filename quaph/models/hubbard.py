import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA

from quaph._model import Model


def _get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _build_H_matrix(lattice, t, U):
    Lx, Ly = lattice
    n_sites = 2 * Lx * Ly
    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)

    def A(i, j): return 2 * (i * Ly + j)
    def B(i, j): return 2 * (i * Ly + j) + 1

    for i in range(Lx):
        for j in range(Ly):
            im = (i - 1) % Lx
            jm = (j - 1) % Ly
            a = A(i, j)
            for b in {B(i, j), B(im, j), B(i, jm)}:
                for s in range(spin):
                    s1, s2 = a * spin + s, b * spin + s
                    H[s1, s2] += -t
                    H[s2, s1] += -t
    return H


def _interaction_hamiltonian(lattice, *, t, U):
    Lx, Ly = lattice
    n_sites = 2 * Lx * Ly
    spin = 2
    H = 0.0 * FermionicOp({}, num_spin_orbitals=n_sites * spin)
    for site in range(n_sites):
        spin_up = site * spin + 0
        spin_down = site * spin + 1
        H += FermionicOp(
            {f"+_{spin_up} -_{spin_up} +_{spin_down} -_{spin_down}": U},
            num_spin_orbitals=n_sites * spin,
        )
    return H


def _mean_field_correction(lattice, n_occ, **params):
    U = params['U']
    if U == 0:
        return 0.0
    Lx, Ly = lattice
    n_sites = 2 * Lx * Ly
    return U * n_sites * (n_occ / (2 * n_sites)) ** 2


model = Model(
    name="hubbard",
    display_name="Hubbard",
    param_labels={"t": "t", "U": "U"},
    spin=2,
    n_dims=2,
    lattice_shape=("Lx", "Ly"),
    sites_per_cell=2,
    hamiltonian_matrix=_build_H_matrix,
    interaction_hamiltonian=_interaction_hamiltonian,
    get_optimizer=_get_optimizer,
    mean_field_correction=_mean_field_correction,
)
