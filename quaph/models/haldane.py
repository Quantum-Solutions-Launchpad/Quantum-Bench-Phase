import numpy as np
from qiskit_algorithms.optimizers import SPSA

from quaph._model import Model


def _get_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _build_H_matrix(n_sites, t1, t2, phi, M):
    if n_sites % 2:
        raise ValueError(f"Haldane n_sites must be even (honeycomb has 2 atoms/cell); got {n_sites}.")
    n_cells = n_sites // 2
    Lx, Ly = 1, n_cells
    for lx in range(2, n_cells // 2 + 1):
        if n_cells % lx == 0 and n_cells // lx >= 2:
            ly = n_cells // lx
            if ((lx % 3 == 0) + (ly % 3 == 0), -abs(lx - ly)) > ((Lx % 3 == 0) + (Ly % 3 == 0), -abs(Lx - Ly)):
                Lx, Ly = lx, ly
    if min(Lx, Ly) < 2:
        raise ValueError(
            f"Haldane n_sites={n_sites} cannot factor as 2*Lx*Ly with Lx, Ly >= 2. "
            f"Use n_sites = 8, 12, 16, 18, 24, 32, 50, 72, ... (n_sites=18 is the smallest "
            f"size that samples the K-point and shows real Haldane phase structure)."
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


model = Model(
    name="haldane",
    display_name="Haldane",
    default_params={"t1": 1.0, "phi": np.pi / 4, "M": 0.0},
    param_labels={"t1": "t_1", "t2": "t_2", "phi": "\\phi", "M": "M"},
    hamiltonian_matrix=_build_H_matrix,
    get_optimizer=_get_optimizer,
    sweep_defaults={"y": {"param": "t2", "range": (0.0, 1.0, 0.1)}},
)
