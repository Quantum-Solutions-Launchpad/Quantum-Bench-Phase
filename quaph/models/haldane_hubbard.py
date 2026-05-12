import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA

from quaph._model import Model


def _get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _build_H_matrix(n_sites, t1, U, t2, phi, M):
    if n_sites % 2:
        raise ValueError(f"Haldane-Hubbard n_sites must be even (honeycomb has 2 atoms/cell); got {n_sites}.")
    n_cells = n_sites // 2
    Lx, Ly = 1, n_cells
    for lx in range(2, n_cells // 2 + 1):
        if n_cells % lx == 0 and n_cells // lx >= 2:
            ly = n_cells // lx
            if ((lx % 3 == 0) + (ly % 3 == 0), -abs(lx - ly)) > ((Lx % 3 == 0) + (Ly % 3 == 0), -abs(Lx - Ly)):
                Lx, Ly = lx, ly
    if min(Lx, Ly) < 2:
        raise ValueError(
            f"Haldane-Hubbard n_sites={n_sites} cannot factor as 2*Lx*Ly with Lx, Ly >= 2. "
            f"Use n_sites = 8, 12, 16, 18, 24, 32, 50, 72, ... (n_sites=18 minimum for K-point sampling)."
        )

    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)

    def A(i, j): return 2 * (i * Ly + j)
    def B(i, j): return 2 * (i * Ly + j) + 1

    for site in range(n_sites):
        sub = +1 if site % 2 == 0 else -1
        for s in range(spin):
            H[site * spin + s, site * spin + s] += sub * M

    for i in range(Lx):
        for j in range(Ly):
            ip, im = (i + 1) % Lx, (i - 1) % Lx
            jp, jm = (j + 1) % Ly, (j - 1) % Ly
            a = A(i, j)
            for b in {B(i, j), B(im, j), B(i, jm)}:
                for s in range(spin):
                    s1, s2 = a * spin + s, b * spin + s
                    H[s1, s2] += -t1
                    H[s2, s1] += -t1
            for tgt in (A(ip, j), A(im, jp), A(i, jm)):
                for s in range(spin):
                    s1, s2 = a * spin + s, tgt * spin + s
                    H[s1, s2] += -t2 * np.exp(+1j * phi)
                    H[s2, s1] += -t2 * np.exp(-1j * phi)
            b0 = B(i, j)
            for tgt in (B(im, j), B(ip, jm), B(i, jp)):
                for s in range(spin):
                    s1, s2 = b0 * spin + s, tgt * spin + s
                    H[s1, s2] += -t2 * np.exp(-1j * phi)
                    H[s2, s1] += -t2 * np.exp(+1j * phi)
    return H


def _fermionic_hamiltonian(n_sites, *, t1, U, t2, phi, M):
    if n_sites % 2:
        raise ValueError(f"Haldane-Hubbard n_sites must be even (honeycomb has 2 atoms/cell); got {n_sites}.")
    n_cells = n_sites // 2
    Lx, Ly = 1, n_cells
    for lx in range(2, n_cells // 2 + 1):
        if n_cells % lx == 0 and n_cells // lx >= 2:
            ly = n_cells // lx
            if ((lx % 3 == 0) + (ly % 3 == 0), -abs(lx - ly)) > ((Lx % 3 == 0) + (Ly % 3 == 0), -abs(Lx - Ly)):
                Lx, Ly = lx, ly
    if min(Lx, Ly) < 2:
        raise ValueError(
            f"Haldane-Hubbard n_sites={n_sites} cannot factor as 2*Lx*Ly with Lx, Ly >= 2."
        )

    spin = 2
    H = 0.0 * FermionicOp({})

    def A(i, j): return 2 * (i * Ly + j)
    def B(i, j): return 2 * (i * Ly + j) + 1

    for site in range(n_sites):
        sub = +1 if site % 2 == 0 else -1
        for s in range(spin):
            idx = site * spin + s
            H += FermionicOp({f"+_{idx} -_{idx}": sub * M})

    for i in range(Lx):
        for j in range(Ly):
            ip, im = (i + 1) % Lx, (i - 1) % Lx
            jp, jm = (j + 1) % Ly, (j - 1) % Ly
            a = A(i, j)
            for b in {B(i, j), B(im, j), B(i, jm)}:
                for s in range(spin):
                    s1, s2 = a * spin + s, b * spin + s
                    H -= FermionicOp({f"+_{s1} -_{s2}": t1, f"+_{s2} -_{s1}": t1})
            for tgt in (A(ip, j), A(im, jp), A(i, jm)):
                for s in range(spin):
                    s1, s2 = a * spin + s, tgt * spin + s
                    H -= FermionicOp({
                        f"+_{s1} -_{s2}": t2 * np.exp(+1j * phi),
                        f"+_{s2} -_{s1}": t2 * np.exp(-1j * phi),
                    })
            b0 = B(i, j)
            for tgt in (B(im, j), B(ip, jm), B(i, jp)):
                for s in range(spin):
                    s1, s2 = b0 * spin + s, tgt * spin + s
                    H -= FermionicOp({
                        f"+_{s1} -_{s2}": t2 * np.exp(-1j * phi),
                        f"+_{s2} -_{s1}": t2 * np.exp(+1j * phi),
                    })

    for site in range(n_sites):
        spin_up = site * spin + 0
        spin_down = site * spin + 1
        H += FermionicOp({
            f"+_{spin_up} -_{spin_up} +_{spin_down} -_{spin_down}": U
        })

    return H


def _mean_field_correction(n_sites, n_occ, **params):
    U = params['U']
    if U == 0:
        return 0.0
    return U * n_sites * (n_occ / (2 * n_sites)) ** 2


model = Model(
    name="haldane-hubbard",
    display_name="Haldane–Hubbard",
    default_params={"t1": 1.0, "phi": np.pi / 4, "M": 0.0},
    param_labels={"t1": "t_1", "U": "U", "t2": "t_2", "phi": "\\phi", "M": "M"},
    hamiltonian_matrix=_build_H_matrix,
    fermionic_hamiltonian=_fermionic_hamiltonian,
    get_optimizer=_get_optimizer,
    mean_field_correction=_mean_field_correction,
    sweep_defaults={
        "x": {"param": "t2", "range": (0.0, 1.5, 0.1)},
        "y": {"param": "U", "range": (0.0, 4.0, 0.5)},
    },
)
