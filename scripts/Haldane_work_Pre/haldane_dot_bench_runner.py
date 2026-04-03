#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from matplotlib.backends.backend_pdf import PdfPages
import glob
import datetime

# ----------------------- Geometry (honeycomb) -----------------------
d1 = np.array([0.0, 1.0])
d2 = np.array([np.sqrt(3) / 2, -0.5])
d3 = np.array([-np.sqrt(3) / 2, -0.5])
NN_VECS = [d1, d2, d3]

a1 = d2 - d3
a2 = d1 - d3

b1 = d2 - d3
b2 = d3 - d1
b3 = d1 - d2
NNN_VECS = [b1, b2, b3]


def _key(pt, ndp=6):
    return (round(pt[0], ndp), round(pt[1], ndp))


# (deprecated placeholder; use generate_sites instead)
# def generate_dot_sites(R: float) -> Tuple[np.ndarray, np.ndarray, dict]:
#     pass
    """Positions (pos), sublattice labels (+1 A, -1 B), and index map for all sites in a disk of radius R."""
    L = np.linalg.norm(a1)
    Nrange = int(np.ceil((R + 2.0) / L)) * 2 + 1

    A_pos, B_pos = [], []
    for n1 in range(-Nrange, Nrange + 1):
        for n2 in range(-Nrange, Nrange + 1):
            Rvec = n1 * a1 + n2 * a2
            rA = Rvec
            rB = Rvec + d1
            if np.linalg.norm(rA) <= R:
                A_pos.append(rA)
            if np.linalg.norm(rB) <= R:
                B_pos.append(rB)

    pos = []
    sub = []
    idx = {}
    for r in A_pos:
        k = _key(r)
        idx[k] = len(pos)
        pos.append(r)
        sub.append(+1)
    for r in B_pos:
        k = _key(r)
        idx[k] = len(pos)
        pos.append(r)
        sub.append(-1)
    return np.array(pos), np.array(sub, dtype=int), idx


def angle(vec):
    return np.arctan2(vec[1], vec[0])


def peierls_phase(r_i, r_j, phi_flux):
    if phi_flux == 0.0:
        return 1.0 + 0j
    theta_i = angle(r_i)
    theta_j = angle(r_j)
    dtheta = np.unwrap([theta_i, theta_j])[1] - np.unwrap([theta_i, theta_j])[0]
    return np.exp(1j * phi_flux * dtheta)

# ---- New geometry helpers and unified site generator ----

def regular_polygon(n: int, R: float, rotation: float = 0.0) -> np.ndarray:
    angles = rotation + 2 * np.pi * np.arange(n) / n
    return np.column_stack([R * np.cos(angles), R * np.sin(angles)])


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-16)
            if x < x_intersect:
                inside = not inside
    return inside


def generate_sites(R: float, shape: str = "disk") -> Tuple[np.ndarray, np.ndarray, dict]:
    L = np.linalg.norm(a1)
    Nrange = int(np.ceil((R + 2.0) / L)) * 2 + 1

    poly = None
    if shape == "hex":
        poly = regular_polygon(6, R, rotation=np.pi / 6)
    elif shape == "triangle":
        poly = regular_polygon(3, R, rotation=np.pi / 2)

    def inside(r):
        if shape == "disk":
            return np.linalg.norm(r) <= R
        else:
            return point_in_polygon(r, poly)

    A_pos, B_pos = [], []
    for n1 in range(-Nrange, Nrange + 1):
            Rvec = n1 * a1 + n2 * a2
            rA = Rvec
            rB = Rvec + d1
            if inside(rA):
                A_pos.append(rA)
            if inside(rB):
                B_pos.append(rB)

    pos, sub, idx = [], [], {}
    for r in A_pos:
        k = _key(r)
        idx[k] = len(pos)
        pos.append(r)
        sub.append(+1)
    for r in B_pos:
        k = _key(r)
        idx[k] = len(pos)
        pos.append(r)
        sub.append(-1)
    return np.array(pos), np.array(sub, dtype=int), idx


# ----------------------- Hamiltonian assembly -----------------------

def build_haldane_hamiltonian(pos, sub, idx_map, t, t2, phi_haldane, M, phi_flux=0.0):
    N = len(pos)
    rows, cols, data = [], [], []

    # On-site Semenoff mass
    for i in range(N):
        rows.append(i)
        cols.append(i)
        data.append(M if sub[i] == +1 else -M)

    # NN hoppings (A->B only, add Hermitian)
    for i in range(N):
        if sub[i] != +1:
            continue
        rA = pos[i]
        for d in NN_VECS:
            rB = rA + d
            j = idx_map.get(_key(rB))
            if j is not None and sub[j] == -1:
                phase = peierls_phase(rA, rB, phi_flux)
                val = t * phase
                rows += [i, j]
                cols += [j, i]
                data += [val, np.conj(val)]

    # NNN hoppings with Haldane phase (avoid double counting with i<j)
    eiphi_A = np.exp(1j * phi_haldane)
    eiphi_B = np.exp(-1j * phi_haldane)
    for i in range(N):
        r = pos[i]
        base_phase = eiphi_A if sub[i] == +1 else eiphi_B
        for b in NNN_VECS:
            r2 = r + b
            j = idx_map.get(_key(r2))
            if j is not None and sub[j] == sub[i] and j > i:
                phase_peierls = peierls_phase(r, r2, phi_flux)
                val = t2 * base_phase * phase_peierls
                rows += [i, j]
                cols += [j, i]
                data += [val, np.conj(val)]

    H = coo_matrix((np.array(data, dtype=np.complex128), (rows, cols)), shape=(N, N)).tocsr()
    return H


# ----------------------- Solvers & metrics -----------------------

def solve_near_zero(H, k, sigma=0.0):
    E, V = eigsh(H, k=k, sigma=sigma, which="LM")
    idx = np.argsort(E)
    return E[idx], V[:, idx]


def edge_participation(V, pos, R, edge_frac):
    r = np.linalg.norm(pos, axis=1)
    rim = r >= R * (1.0 - edge_frac)
    epr = []
    for j in range(V.shape[1]):
        psi2 = np.abs(V[:, j]) ** 2
        epr.append(psi2[rim].sum() / psi2.sum())
    return np.array(epr)


def pick_edge_state(E, epr):
    i0 = np.argmin(np.abs(E))
    window = slice(max(i0 - 10, 0), min(i0 + 11, len(E)))
    subidx = np.arange(len(E))[window]
    best = subidx[np.argmax(epr[window])]
    return best


def level_spacing_from_edge_ladder(E, epr, epr_thresh=0.5, center_window=0.6):
    mask_edge = epr >= epr_thresh
    if not np.any(mask_edge):
        return np.nan
    E_edge = E[mask_edge]
    E_mid = E_edge[np.abs(E_edge) <= center_window * np.max(np.abs(E_edge))]
    if len(E_mid) < 3:
        return np.nan
    E_mid = np.sort(E_mid)
    d = np.diff(E_mid)
    return np.median(d)


def ipr_all(V):
    return np.sum(np.abs(V) ** 4, axis=0)


def hermiticity_error(H):
    diff = H - H.getH()
    num = np.linalg.norm(diff.toarray())
    den = np.linalg.norm(H.toarray())
    return num / max(den, 1e-16)


# ----------------------- Plots -----------------------

def plot_spectrum(E, epr, title, outpath=None):
    plt.figure(figsize=(6.4, 3.9))
    sc = plt.scatter(np.arange(len(E)), E, c=epr, s=12, cmap="viridis")
    plt.axhline(0.0, ls="--", lw=0.8, c="k")
    cb = plt.colorbar(sc)
    cb.set_label("edge participation ratio")
    plt.xlabel("eigenstate index (sorted)")
    plt.ylabel("energy")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_ldos(pos, psi, R, Eval, title_prefix, outpath=None):
    psi2 = np.abs(psi) ** 2
    plt.figure(figsize=(5.6, 4.9))
    plt.scatter(pos[:, 0], pos[:, 1], c=psi2 / psi2.max(), s=18, cmap="viridis")
    circ = plt.Circle((0, 0), R, fill=False, ls="--", lw=0.8, color="k")
    ax = plt.gca()
    ax.add_artist(circ)
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix} @ E={Eval:.4f}")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_spacing_vs_invR(R_list, Delta_list, v_edge_fit=None, outpath=None):
    invR = 1.0 / np.array(R_list, dtype=float)
    plt.figure(figsize=(5.9, 3.9))
    plt.scatter(invR, Delta_list, s=28)
    if v_edge_fit is not None and np.isfinite(v_edge_fit):
        xs = np.linspace(invR.min() * 0.9, invR.max() * 1.05, 100)
        plt.plot(xs, v_edge_fit * xs, lw=1.5)
        plt.legend([r"$\Delta$ vs $1/R$", rf"fit: $\Delta \approx {v_edge_fit:.3f}/R$"]) 
    else:
        plt.legend([r"$\Delta$ vs $1/R$"])
    plt.xlabel(r"$1/R$")
    plt.ylabel(r"edge level spacing $\Delta$")
    plt.title("Edge spacing scales ~ $v_{\mathrm{edge}}/R$")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_spectral_flow(phi_list, E_flow, outpath=None):
    plt.figure(figsize=(6.6, 4.0))
    for row in E_flow:
        plt.plot(phi_list, row, lw=1.0)
    plt.xlabel(r"flux $\Phi/\Phi_0$")
    plt.ylabel("energy")
    plt.title("Spectral flow of low-lying levels (edge states move across gap)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_edge_currents(pos, H, psi, R=None, edge_frac=0.2, mag_clip=0.95, outpath=None):
    rows, cols = H.nonzero()
    js = []
    xs, ys, us, vs = [], [], [], []
    rim_mask = None
    if R is not None:
        rim_start = R * (1.0 - edge_frac)
        # later we check bonds with both endpoints in rim
    rim_currents = []

    for i, j in zip(rows, cols):
        if i >= j:
            continue
        Hij = H[i, j]
        if Hij == 0:
            continue
        j_ij = -2.0 * np.imag(Hij * np.conj(psi[i]) * psi[j])
        if j_ij == 0:
            continue
        ri = pos[i]
        rj = pos[j]
        if R is not None:
            if (np.linalg.norm(ri) >= rim_start) and (np.linalg.norm(rj) >= rim_start):
                rim_currents.append(abs(j_ij))
        mid = 0.5 * (ri + rj)
        direction = rj - ri
        norm = np.linalg.norm(direction) + 1e-12
        direction = direction / norm
        xs.append(mid[0])
        ys.append(mid[1])
        us.append(direction[0] * j_ij)
        vs.append(direction[1] * j_ij)
        js.append(abs(j_ij))

    mean_rim_current = float(np.mean(rim_currents)) if len(rim_currents) else 0.0

    if not js:
        return mean_rim_current

    js = np.array(js)
    scale = np.quantile(js, mag_clip)
    us = np.array(us) / (scale + 1e-12)
    vs = np.array(vs) / (scale + 1e-12)

    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(pos[:, 0], pos[:, 1], s=6, c="gray", alpha=0.3)
    plt.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1, width=0.003)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Edge currents (arrows scaled by magnitude)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()

    return mean_rim_current


# ----------------------- Workflows -----------------------
@dataclass
class RunConfig:
    t: float = 1.0
    t2: float = 0.1
    phi: float = np.pi / 2
    M: float = 0.2
    R: float = 15.0
    edge_frac: float = 0.2
    k_eigs: int = 120
    seed: int = 0
    outdir: str = "out"

    # report
    make_report: bool = False
    report_name: str = "report.pdf"

    # geometry/options
    shape: str = "disk"  # one of {"disk","hex","triangle"}

    # tasks
    do_single: bool = True
    do_size_sweep: bool = True
    do_flux: bool = False
    do_disorder: bool = False
    do_phase_sweep: bool = False
    do_currents: bool = False
    do_phi_flip_check: bool = False
    export_vqe_scaffold: bool = False
    vqe_max_sites: int = 24

    # sweeps
    R_list: List[float] = None
    flux_min: float = 0.0
    flux_max: float = 0.5
    flux_points: int = 13

    disorder_W_list: List[float] = None
    disorder_seeds: int = 3

    M_min: float = -0.8
    M_max: float = 0.8
    M_points: int = 21


def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def single_dot_demo(cfg: RunConfig):
    np.random.seed(cfg.seed)
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0)

    h_err = hermiticity_error(H)
    E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
    epr = edge_participation(V, pos, cfg.R, cfg.edge_frac)
    j = pick_edge_state(E, epr)

    plot_spectrum(E, epr, f"Haldane dot R={cfg.R:.1f} (EPR colored)", outpath=os.path.join(cfg.outdir, "spectrum_R%.1f.png" % cfg.R))
    plot_ldos(pos, V[:, j], cfg.R, E[j], title_prefix="Edge-state LDOS", outpath=os.path.join(cfg.outdir, "ldos_R%.1f.png" % cfg.R))

    mean_rim_current = plot_edge_currents(pos, H, V[:, j], R=cfg.R, edge_frac=cfg.edge_frac, outpath=os.path.join(cfg.outdir, "edge_currents.png")) if cfg.do_currents else 0.0

    np.savetxt(os.path.join(cfg.outdir, "eigs.csv"), E)
    np.savetxt(os.path.join(cfg.outdir, "epr.csv"), epr)

    metrics = {
        "hermiticity_error": float(h_err),
        "min_abs_energy": float(np.min(np.abs(E))),
        "num_sites": int(len(pos)),
        "seed": cfg.seed,
        "mean_rim_current": float(mean_rim_current),
        "shape": cfg.shape,
    }
    save_json(metrics, os.path.join(cfg.outdir, "metrics_single.json"))


def size_sweep(cfg: RunConfig):
    R_list = cfg.R_list or [10, 12, 14, 16, 18, 20]
    deltas = []
    records = []
    for R in R_list:
        pos, sub, idx = generate_sites(R, cfg.shape)
        H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0)
        E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
        epr = edge_participation(V, pos, R, cfg.edge_frac)
        Delta = level_spacing_from_edge_ladder(E, epr, epr_thresh=0.5)
        deltas.append(Delta)
        records.append([R, len(pos), Delta])
        print(f"[size] R={R:>4.1f} sites={len(pos):>5d} Δ≈{Delta:.6f}")

    records = np.array(records, dtype=float)
    np.savetxt(os.path.join(cfg.outdir, "size_sweep.csv"), records, delimiter=",", header="R,num_sites,Delta", comments="")

    invR = 1.0 / records[:, 0]
    Delta = records[:, 2]
    mask = np.isfinite(Delta)
    v_edge = np.nan
    r2 = np.nan
    if np.count_nonzero(mask) >= 2:
        coeffs = np.polyfit(invR[mask], Delta[mask], 1)
        v_edge = coeffs[0]
        yhat = np.polyval(coeffs, invR[mask])
        ss_res = np.sum((Delta[mask] - yhat) ** 2)
        ss_tot = np.sum((Delta[mask] - np.mean(Delta[mask])) ** 2) + 1e-16
        r2 = 1 - ss_res / ss_tot
    plot_spacing_vs_invR(records[:, 0], Delta, v_edge_fit=v_edge, outpath=os.path.join(cfg.outdir, "spacing_vs_invR.png"))

    save_json({"v_edge": float(v_edge) if np.isfinite(v_edge) else None, "r2": float(r2) if np.isfinite(r2) else None}, os.path.join(cfg.outdir, "metrics_size.json"))


def flux_flow(cfg: RunConfig):
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    phi_list = np.linspace(cfg.flux_min, cfg.flux_max, cfg.flux_points)
    E_flow = []
    for phi_flux in phi_list:
        H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=phi_flux)
        E, _ = solve_near_zero(H, k=min(cfg.k_eigs, 64), sigma=0.0)
        E_flow.append(E[:32])
        print(f"[flux] Φ/Φ0={phi_flux:.3f} min|E|={np.min(np.abs(E)):.4e}")

    E_flow = np.array(E_flow).T
    np.savetxt(os.path.join(cfg.outdir, "spectral_flow.csv"), E_flow, delimiter=",")
    np.savetxt(os.path.join(cfg.outdir, "spectral_flow_phi.csv"), phi_list)
    plot_spectral_flow(phi_list, E_flow, outpath=os.path.join(cfg.outdir, "spectral_flow.png"))

    return phi_list, E_flow


def add_disorder_on_site(H, W, rng):
    if W <= 0:
        return H
    H2 = H.tolil(copy=True)
    N = H.shape[0]
    diag = H2.diagonal().astype(np.complex128)
    diag += rng.uniform(-W, W, size=N)
    H2.setdiag(diag)
    return H2.tocsr()


def disorder_sweep(cfg: RunConfig):
    W_list = cfg.disorder_W_list or [0.0, 0.3, 0.6, 0.9]
    R = cfg.R
    pos, sub, idx = generate_sites(R, cfg.shape)
    rng_master = np.random.RandomState(cfg.seed)

    results = []
    for W in W_list:
        edge_counts = []
        gaps = []
        for s in range(cfg.disorder_seeds):
            rng = np.random.RandomState(rng_master.randint(0, 2**31 - 1))
            H0 = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0)
            H = add_disorder_on_site(H0, W, rng)
            E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
            epr = edge_participation(V, pos, R, cfg.edge_frac)
            mask_edge = (epr >= 0.5) & (np.abs(E) < 0.6 * np.max(np.abs(E)))
            edge_counts.append(int(np.count_nonzero(mask_edge)))
            pos_min = np.min(E[E > 0]) if np.any(E > 0) else np.nan
            neg_max = np.max(E[E < 0]) if np.any(E < 0) else np.nan
            gap = pos_min - neg_max if np.isfinite(pos_min) and np.isfinite(neg_max) else np.nan
            gaps.append(gap)
        results.append([W, float(np.nanmean(edge_counts)), float(np.nanstd(edge_counts)), float(np.nanmean(gaps))])
        print(f"[disorder] W={W:.2f} edge_count_mean={np.mean(edge_counts):.2f} gap_mean={np.nanmean(gaps):.4f}")

    arr = np.array(results)
    np.savetxt(os.path.join(cfg.outdir, "disorder_sweep.csv"), arr, delimiter=",", header="W,edge_count_mean,edge_count_std,gap_mean", comments="")

    plt.figure(figsize=(6.0, 4.0))
    plt.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], fmt="o-")
    plt.xlabel("W (on-site disorder)")
    plt.ylabel("in-gap edge-state count (mean ± std)")
    plt.title("Edge ladder robustness vs disorder")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "disorder_edgecount.png"), dpi=200)
    plt.close()


def phase_sweep_M(cfg: RunConfig):
    Ms = np.linspace(cfg.M_min, cfg.M_max, cfg.M_points)
    R = cfg.R
    pos, sub, idx = generate_sites(R, cfg.shape)
    records = []
    for M in Ms:
        H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, M, phi_flux=0.0)
        E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
        epr = edge_participation(V, pos, R, cfg.edge_frac)
        mask_edge = (epr >= 0.5) & (np.abs(E) < 0.6 * np.max(np.abs(E)))
        edge_sum = float(np.sum(epr[mask_edge]))
        count = int(np.count_nonzero(mask_edge))
        pos_min = np.min(E[E > 0]) if np.any(E > 0) else np.nan
        neg_max = np.max(E[E < 0]) if np.any(E < 0) else np.nan
        gap = pos_min - neg_max if np.isfinite(pos_min) and np.isfinite(neg_max) else np.nan
        records.append([M, edge_sum, count, gap])
        print(f"[phase] M={M:+.3f} edge_sum={edge_sum:.3f} edge_count={count:3d} gap={gap:.4f}")

    arr = np.array(records)
    np.savetxt(os.path.join(cfg.outdir, "phase_sweep_M.csv"), arr, delimiter=",", header="M,edge_sum,count,gap", comments="")

    Mc = 3 * np.sqrt(3) * cfg.t2 * np.sin(cfg.phi)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax1.plot(arr[:, 0], arr[:, 1], "o-", label="total edge weight (in-gap)")
    ax1.set_xlabel("M")
    ax1.set_ylabel("edge weight (sum of EPR)")
    ax2 = ax1.twinx()
    ax2.plot(arr[:, 0], arr[:, 2], "s-", color="tab:orange", label="edge count")
    ax2.set_ylabel("in-gap edge-state count")
    for Mc_val in [+Mc, -Mc]:
        ax1.axvline(Mc_val, ls="--", color="k", alpha=0.5)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.title("Phase sweep vs M (dashed = predicted transition)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "phase_sweep_M.png"), dpi=200)
    plt.close()


def currents_panel(cfg: RunConfig):
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0)
    E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
    epr = edge_participation(V, pos, cfg.R, cfg.edge_frac)
    j = pick_edge_state(E, epr)
    plot_edge_currents(pos, H, V[:, j], outpath=os.path.join(cfg.outdir, "edge_currents.png"))


# ----------------------- Report bundling (PDF) -----------------------

def _text_block_from_dict(d):
    lines = []
    for k, v in d.items():
        lines.append(f"{k}: {v}")
    return "".join(lines)


def _add_text_page(pdf: PdfPages, title: str, body: str):
    fig = plt.figure(figsize=(8.3, 11.7))  # A4-ish
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')
    ax.text(0.0, 1.0, title, fontsize=16, weight='bold', va='top')
    ax.text(0.0, 0.95, body, fontsize=10, va='top', family='monospace')
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, img_path: str, title: str = None):
    if not os.path.exists(img_path):
        return
    img = plt.imread(img_path)
    fig = plt.figure(figsize=(8.3, 11.7))
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.85])
    ax.imshow(img)
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def bundle_report_pdf(cfg):
    out_pdf = os.path.join(cfg.outdir, cfg.report_name)
    with PdfPages(out_pdf) as pdf:
        # Cover page
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cover_title = "Haldane Quantum Dot - Benchmark Report"
        body = [
            f"Generated: {now}",
            "",
            "Run configuration:",
            _text_block_from_dict({
                't': cfg.t, 't2': cfg.t2, 'phi': cfg.phi, 'M': cfg.M, 'R': cfg.R,
                'edge_frac': cfg.edge_frac, 'k_eigs': cfg.k_eigs, 'seed': cfg.seed
            })
        ]
        _add_text_page(pdf, cover_title, "".join(body))

        # Metrics summary pages (if present)
        single_metrics = os.path.join(cfg.outdir, 'metrics_single.json')
        size_metrics = os.path.join(cfg.outdir, 'metrics_size.json')
        if os.path.exists(single_metrics):
            with open(single_metrics, 'r') as f:
                m = json.load(f)
            _add_text_page(pdf, 'Single-dot metrics', _text_block_from_dict(m))
        if os.path.exists(size_metrics):
            with open(size_metrics, 'r') as f:
                m = json.load(f)
            _add_text_page(pdf, 'Size-sweep fit metrics', _text_block_from_dict(m))

        # Add figures in a sensible order if they exist
        fig_order = [
            os.path.join(cfg.outdir, f'spectrum_R{cfg.R:.1f}.png'),
            os.path.join(cfg.outdir, f'ldos_R{cfg.R:.1f}.png'),
            os.path.join(cfg.outdir, 'spacing_vs_invR.png'),
            os.path.join(cfg.outdir, 'spectral_flow.png'),
            os.path.join(cfg.outdir, 'edge_currents.png'),
            os.path.join(cfg.outdir, 'disorder_edgecount.png'),
            os.path.join(cfg.outdir, 'phase_sweep_M.png'),
        ]
        titles = [
            'Spectrum (EPR-colored)',
            'LDOS of representative edge state',
            'Edge spacing vs 1/R (fit)',
            'Spectral flow under flux',
            'Edge current map',
            'Edge ladder robustness vs disorder',
            'Phase sweep vs M',
        ]
        for pth, ttl in zip(fig_order, titles):
            _add_image_page(pdf, pth, ttl)

    print(f"[report] Wrote {out_pdf}")

# ----------------------- Extras -----------------------

def phi_flip_check(cfg: RunConfig):
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    phi_list = np.linspace(cfg.flux_min, cfg.flux_max, cfg.flux_points)

    def spectral_flow_for_phi(phi):
        E_flow = []
        for phi_flux in phi_list:
            H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, phi, cfg.M, phi_flux=phi_flux)
            E, _ = solve_near_zero(H, k=min(cfg.k_eigs, 64), sigma=0.0)
            E_flow.append(E[:24])
        return np.array(E_flow).T

    E_plus = spectral_flow_for_phi(cfg.phi)
    E_minus = spectral_flow_for_phi(-cfg.phi)

    # quick slope estimate for the first K levels (by |E|)
    def slope_stats(E_flow):
        K = min(12, E_flow.shape[0])
        idxs = np.argsort(np.min(np.abs(E_flow), axis=1))[:K]
        dphi = phi_list[-1] - phi_list[0] + 1e-12
        slopes = (E_flow[idxs, -1] - E_flow[idxs, 0]) / dphi
        return float(np.mean(slopes)), float(np.std(slopes))

    m_plus, s_plus = slope_stats(E_plus)
    m_minus, s_minus = slope_stats(E_minus)

    # plot overlay
    plt.figure(figsize=(6.8, 4.2))
    for row in E_plus:
        plt.plot(phi_list, row, lw=1.0, alpha=0.9, label='+phi' if 'plotted_plus' not in locals() else None)
        plotted_plus = True
    for row in E_minus:
        plt.plot(phi_list, row, lw=1.0, alpha=0.7, linestyle='--', label='-phi' if 'plotted_minus' not in locals() else None)
        plotted_minus = True
    plt.xlabel('flux (Phi/Phi0)')
    plt.ylabel('energy')
    plt.title('Spectral flow: +phi (solid) vs -phi (dashed)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, 'spectral_flow_phi_flip.png'), dpi=200)
    plt.close()

    save_json({
        'mean_slope_plus_phi': m_plus,
        'std_slope_plus_phi': s_plus,
        'mean_slope_minus_phi': m_minus,
        'std_slope_minus_phi': s_minus
    }, os.path.join(cfg.outdir, 'metrics_phi_flip.json'))


def export_vqe_scaffold(cfg: RunConfig):
    # Build a tiny dot and export second-quantized hopping terms for future VQE mapping
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    if len(pos) > cfg.vqe_max_sites:
        # shrink R until under the cap
        R_try = cfg.R
        while len(pos) > cfg.vqe_max_sites and R_try > 2.0:
            R_try *= 0.9
            pos, sub, idx = generate_sites(R_try, cfg.shape)
        print(f"[vqe] resized dot to R≈{R_try:.2f} for <= {cfg.vqe_max_sites} sites (actual {len(pos)})")
    H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0).tocsr()

    # Export on-site and hopping terms: i,j, Re, Im
    rows, cols = H.nonzero()
    terms = []
    for i, j in zip(rows, cols):
        if abs(H[i, j]) == 0:
            continue
        terms.append([int(i), int(j), float(np.real(H[i, j])), float(np.imag(H[i, j]))])
    terms = np.array(terms)
    np.savetxt(os.path.join(cfg.outdir, 'vqe_terms_ij_ReIm.csv'), terms, delimiter=',', header='i,j,Re,Im', comments='')
    np.savetxt(os.path.join(cfg.outdir, 'vqe_positions_xy.csv'), pos, delimiter=',', header='x,y', comments='')
    np.savetxt(os.path.join(cfg.outdir, 'vqe_sublattice_pm1.csv'), sub, delimiter=',', header='sublattice(+1=A,-1=B)', comments='')

    with open(os.path.join(cfg.outdir, 'vqe_README.txt'), 'w') as f:
        f.write(
            'This folder contains a small Haldane-dot instance exported for VQE scaffolding.'
            'Files:'
            '  - vqe_terms_ij_ReIm.csv : rows of i,j,Re(H_ij),Im(H_ij) for the single-particle Hamiltonian.'
            '  - vqe_positions_xy.csv  : site coordinates (for reference/plotting).'
            '  - vqe_sublattice_pm1.csv: +1 for A, -1 for B.'
            ' Use your preferred fermion-to-qubit mapping (e.g., Jordan-Wigner) to build a qubit Hamiltonian.'
        )
    print('[vqe] Exported scaffold files for VQE in', cfg.outdir)

# ----------------------- CLI -----------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Haldane quantum-dot benchmark runner")
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--t2", type=float, default=0.1)
    p.add_argument("--phi", type=float, default=np.pi / 2)
    p.add_argument("--M", type=float, default=0.2)
    p.add_argument("--R", type=float, default=15.0)
    p.add_argument("--edge_frac", type=float, default=0.2)
    p.add_argument("--k_eigs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="out")

    p.add_argument("--do_single", action="store_true")
    p.add_argument("--do_size_sweep", action="store_true")
    p.add_argument("--do_flux", action="store_true")
    p.add_argument("--do_disorder", action="store_true")
    p.add_argument("--do_phase_sweep", action="store_true")
    p.add_argument("--do_currents", action="store_true")

    # geometry
    p.add_argument("--shape", type=str, default="disk", choices=["disk", "hex", "triangle"]) 

    # report
    p.add_argument("--make_report", action="store_true")
    p.add_argument("--report_name", type=str, default="report.pdf")

    # extras
    p.add_argument("--do_phi_flip_check", action="store_true")
    p.add_argument("--export_vqe_scaffold", action="store_true")
    p.add_argument("--vqe_max_sites", type=int, default=24)

    p.add_argument("--R_list", type=str, default="10,12,14,16,18,20")
    p.add_argument("--flux_min", type=float, default=0.0)
    p.add_argument("--flux_max", type=float, default=0.5)
    p.add_argument("--flux_points", type=int, default=13)

    p.add_argument("--disorder_W_list", type=str, default="0.0,0.3,0.6,0.9")
    p.add_argument("--disorder_seeds", type=int, default=3)

    p.add_argument("--M_min", type=float, default=-0.8)
    p.add_argument("--M_max", type=float, default=0.8)
    p.add_argument("--M_points", type=int, default=21)

    args = p.parse_args()

    cfg = RunConfig()
    cfg.t = args.t
    cfg.t2 = args.t2
    cfg.phi = args.phi
    cfg.M = args.M
    cfg.R = args.R
    cfg.edge_frac = args.edge_frac
    cfg.k_eigs = args.k_eigs
    cfg.seed = args.seed
    cfg.outdir = args.outdir

    cfg.make_report = args.make_report
    cfg.report_name = args.report_name

    cfg.shape = args.shape
    cfg.do_phi_flip_check = args.do_phi_flip_check
    cfg.export_vqe_scaffold = args.export_vqe_scaffold
    cfg.vqe_max_sites = args.vqe_max_sites

    cfg.do_single = args.do_single
    cfg.do_size_sweep = args.do_size_sweep
    cfg.do_flux = args.do_flux
    cfg.do_disorder = args.do_disorder
    cfg.do_phase_sweep = args.do_phase_sweep
    cfg.do_currents = args.do_currents

    cfg.R_list = [float(x) for x in args.R_list.split(",") if x.strip()]
    cfg.flux_min = args.flux_min
    cfg.flux_max = args.flux_max
    cfg.flux_points = args.flux_points

    cfg.disorder_W_list = [float(x) for x in args.disorder_W_list.split(",") if x.strip()]
    cfg.disorder_seeds = args.disorder_seeds

    cfg.M_min = args.M_min
    cfg.M_max = args.M_max
    cfg.M_points = args.M_points

    return cfg


def main():
    cfg = parse_args()
    ensure_outdir(cfg.outdir)

    # Save run config
    save_json(asdict(cfg), os.path.join(cfg.outdir, "run_config.json"))

    # Single-dot baseline
    if cfg.do_single:
        single_dot_demo(cfg)

    # Size scaling
    if cfg.do_size_sweep:
        size_sweep(cfg)

    # Flux spectral flow
    if cfg.do_flux:
        flux_flow(cfg)

    # Disorder robustness
    if cfg.do_disorder:
        disorder_sweep(cfg)

    # Phase sweep in M
    if cfg.do_phase_sweep:
        phase_sweep_M(cfg)

    # Edge currents
    if cfg.do_currents:
        currents_panel(cfg)

    # phi flip check
    if cfg.do_phi_flip_check:
        phi_flip_check(cfg)

    # VQE scaffold export
    if cfg.export_vqe_scaffold:
        export_vqe_scaffold(cfg)

    # Bundle report
    if cfg.make_report:
        bundle_report_pdf(cfg)


if __name__ == "__main__":
    main()
