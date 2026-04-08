import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA
from loguru import logger


NAME = "hubbard"
DISPLAY_NAME = "Hubbard"
DEFAULT_PARAMS = {"t": 1.0, "U": 0.0}
PARAM_LABELS = {"t": "t", "U": "U"}

def get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)

def file_suffix(params):
    return f"U-{params['U']}"

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

def fermionic_hamiltonian(n_sites, *, t, U):
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

def real_space_exact(n_sites, n_occ, *, t, U):
    H = _build_H_matrix(n_sites, t, U)
    eigvals, _ = np.linalg.eigh(H)
    kinetic_energy = np.sum(np.sort(eigvals)[:n_occ])

    if U != 0:
        avg_double_occupancy = (n_occ / (2 * n_sites)) ** 2
        interaction_energy = U * n_sites * avg_double_occupancy
    else:
        interaction_energy = 0.0

    result = kinetic_energy + interaction_energy
    logger.info(f"Exact (n_sites={n_sites}, n_occ={n_occ}, t={t}, U={U}) = {result}")
    return result
