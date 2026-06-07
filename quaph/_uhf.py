from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Seed = Literal["paramagnetic", "neel", "ferromagnetic"]


@dataclass
class UHFResult:
    energy: float
    n_up: np.ndarray
    n_dn: np.ndarray
    eigvals_up: np.ndarray
    eigvals_dn: np.ndarray
    occ_up_idx: list[int]
    occ_dn_idx: list[int]
    converged: bool
    iterations: int
    seed: Seed


def _initial_densities(
    n_sites: int,
    n_occ: int,
    sublattice_of_site,
    seed: Seed,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    rho = n_occ / (2 * n_sites)
    n_up = np.full(n_sites, rho, dtype=float)
    n_dn = np.full(n_sites, rho, dtype=float)
    if seed == "paramagnetic":
        return n_up, n_dn
    delta = min(amplitude, rho, 1.0 - rho)
    if seed == "ferromagnetic":
        return n_up + delta, n_dn - delta
    if seed == "neel":
        for s in range(n_sites):
            sign = +1 if sublattice_of_site(s) % 2 == 0 else -1
            n_up[s] += sign * delta
            n_dn[s] -= sign * delta
        return n_up, n_dn
    raise ValueError(f"unknown seed '{seed}'")


def _occupy_with_spin_balance(eigvals_up, eigvals_dn, n_occ_total):
    levels = []
    for i, e in enumerate(eigvals_up):
        levels.append((float(e), 0, i))
    for i, e in enumerate(eigvals_dn):
        levels.append((float(e), 1, i))
    levels.sort()
    occ_up: list[int] = []
    occ_dn: list[int] = []
    for e, s, i in levels[:n_occ_total]:
        if s == 0:
            occ_up.append(i)
        else:
            occ_dn.append(i)
    return occ_up, occ_dn


def run_uhf(
    model,
    lattice,
    n_occ: int,
    params: dict,
    U: float,
    *,
    seed: Seed = "paramagnetic",
    amplitude: float = 0.5,
    max_iter: int = 400,
    tol: float = 1e-7,
    mix: float = 0.3,
) -> UHFResult:
    lat = tuple(lattice)
    n_cells = 1
    for L in lat:
        n_cells *= L
    sites_per_cell = model.sites_per_cell
    n_sites = n_cells * sites_per_cell
    spin = model.spin
    if spin != 2:
        raise ValueError("UHF requires spin=2")

    H_full = model._build_H_matrix(lat, **params)
    H_up = H_full[0::2, 0::2].copy()
    H_dn = H_full[1::2, 1::2].copy()

    def sublattice_of_site(s: int) -> int:
        return s % sites_per_cell

    n_up, n_dn = _initial_densities(
        n_sites, n_occ, sublattice_of_site, seed, amplitude
    )

    prev_E = None
    converged = False
    E = float("nan")
    eu = ed = np.array([])
    occ_up_idx: list[int] = []
    occ_dn_idx: list[int] = []
    for it in range(1, max_iter + 1):
        H_up_eff = H_up + np.diag(U * n_dn).astype(complex)
        H_dn_eff = H_dn + np.diag(U * n_up).astype(complex)
        eu, Vu = np.linalg.eigh(H_up_eff)
        ed, Vd = np.linalg.eigh(H_dn_eff)
        occ_up_idx, occ_dn_idx = _occupy_with_spin_balance(eu, ed, n_occ)
        Vu_occ = Vu[:, occ_up_idx]
        Vd_occ = Vd[:, occ_dn_idx]
        new_n_up = np.real(np.einsum("ij,ij->i", Vu_occ.conj(), Vu_occ))
        new_n_dn = np.real(np.einsum("ij,ij->i", Vd_occ.conj(), Vd_occ))

        n_up = (1 - mix) * n_up + mix * new_n_up
        n_dn = (1 - mix) * n_dn + mix * new_n_dn

        band_E = float(sum(eu[i] for i in occ_up_idx) + sum(ed[i] for i in occ_dn_idx))
        dc = float(U * np.sum(n_up * n_dn))
        E = band_E - dc
        if prev_E is not None and abs(E - prev_E) < tol:
            converged = True
            break
        prev_E = E

    return UHFResult(
        energy=float(E),
        n_up=n_up,
        n_dn=n_dn,
        eigvals_up=eu,
        eigvals_dn=ed,
        occ_up_idx=occ_up_idx,
        occ_dn_idx=occ_dn_idx,
        converged=converged,
        iterations=it,
        seed=seed,
    )


def run_uhf_lowest(
    model,
    lattice,
    n_occ: int,
    params: dict,
    U: float,
    *,
    seeds: tuple[Seed, ...] = ("paramagnetic", "neel", "ferromagnetic"),
    **kwargs,
) -> UHFResult:
    best: UHFResult | None = None
    for s in seeds:
        r = run_uhf(model, lattice, n_occ, params, U, seed=s, **kwargs)
        if best is None or r.energy < best.energy:
            best = r
    assert best is not None
    return best


def staggered_magnetization(result: UHFResult, sites_per_cell: int) -> float:
    m_local = result.n_up - result.n_dn
    signs = np.array(
        [+1 if (s % sites_per_cell) % 2 == 0 else -1 for s in range(len(m_local))]
    )
    return float(np.sum(m_local * signs) / len(m_local))


def total_magnetization(result: UHFResult) -> float:
    return float(np.sum(result.n_up - result.n_dn) / len(result.n_up))


def hf_gap(result: UHFResult, n_occ: int) -> float:
    e_all = np.sort(np.concatenate([result.eigvals_up, result.eigvals_dn]))
    if n_occ <= 0 or n_occ >= len(e_all):
        return 0.0
    return float(e_all[n_occ] - e_all[n_occ - 1])
