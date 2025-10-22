# ==============================================
# File: haldane_dot_bench_runner.py
# ==============================================

from __future__ import annotations
import os, json, math, glob, datetime
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------- Lattice geometry -----------------------
# Honeycomb vectors (a = 1):
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])
# A at (0,0); B at d1
# choose one NN displacement from A to B
d1 = np.array([0.0, 0.0]) + (a1 + a2) / 3.0  # conventional choice

# NN displacements from A to B
NN_AB = [
    d1,
    d1 - a1,
    d1 - a2,
]
# NNN displacements within same sublattice (three vectors)
NNN_AA = [a1, a2, a2 - a1]


def angle(r: np.ndarray) -> float:
    return float(np.arctan2(r[1], r[0]))


def _key(pt, ndp=6):
    return (round(float(pt[0]), ndp), round(float(pt[1]), ndp))


def regular_polygon(n: int, R: float, rotation: float = 0.0) -> np.ndarray:
    ang = rotation + 2 * np.pi * (np.arange(n) / n)
    return np.column_stack([R * np.cos(ang), R * np.sin(ang)])


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # Check if edge crosses the horizontal ray
        if ((y1 > y) != (y2 > y)):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-16)
            if x < x_int:
                inside = not inside
    return inside


def generate_sites(R: float, shape: str = "disk") -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (positions, sublattice labels (+1 A / -1 B), index dict)."""
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
        for n2 in range(-Nrange, Nrange + 1):
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
        if k in idx:
            continue
        idx[k] = len(pos)
        pos.append(r)
        sub.append(-1)

    return np.array(pos, dtype=float), np.array(sub, dtype=int), idx


# ----------------------- Haldane Hamiltonian -----------------------

def peierls_phase(r_i, r_j, phi_flux):
    if phi_flux == 0.0:
        return 1.0 + 0j
    # Use a simple polar-angle gauge so a loop gains 2πΦ/Φ0
    th_i = angle(r_i)
    th_j = angle(r_j)
    dth = np.unwrap([th_i, th_j])
    dtheta = dth[1] - dth[0]
    return np.exp(1j * phi_flux * dtheta)


def build_haldane_hamiltonian(pos: np.ndarray, sub: np.ndarray, idx: dict,
                               t: float, t2: float, phi: float, M: float,
                               phi_flux: float = 0.0,
                               V_soft: np.ndarray | None = None) -> csr_matrix:
    N = len(pos)
    rows, cols, data = [], [], []

    # On-site Semenoff mass and optional soft wall
    for i in range(N):
        rows.append(i); cols.append(i); data.append(M if sub[i] == +1 else -M)
        if V_soft is not None:
            rows.append(i); cols.append(i); data.append(float(V_soft[i]))

    # NN hoppings A<->B
    for kA, rA in enumerate(pos[sub == +1]):
        i = np.where((pos == rA).all(axis=1))[0][0]
        for disp in NN_AB:
            rB = rA + disp
            j = idx.get(_key(rB))
            if j is None:
                continue
            phase = peierls_phase(rA, rB, phi_flux)
            rows += [i]; cols += [j]; data += [t * phase]
            rows += [j]; cols += [i]; data += [t * np.conj(phase)]

    # NNN hoppings within same sublattice (Haldane term)
    for i in range(N):
        r_i = pos[i]
        eiphi = np.exp(1j * phi) if sub[i] == +1 else np.exp(-1j * phi)
        for disp in NNN_AA:
            r_j = r_i + disp
            j = idx.get(_key(r_j))
            if j is None or j == i:
                continue
            phase = peierls_phase(r_i, r_j, phi_flux)
            amp = t2 * eiphi * phase
            rows += [i]; cols += [j]; data += [amp]
            rows += [j]; cols += [i]; data += [np.conj(amp)]

    H = coo_matrix((data, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    return H


# ----------------------- Solvers & metrics -----------------------

def hermiticity_error(H: csr_matrix) -> float:
    diff = H - H.getH()
    num = np.linalg.norm(diff.data) if diff.nnz else 0.0
    den = np.linalg.norm(H.data) if H.nnz else 1.0
    return float(num / den)


def solve_near_zero(H: csr_matrix, k: int = 100, sigma: float = 0.0):
    k = min(k, H.shape[0]-2)
    vals, vecs = eigsh(H, k=k, sigma=sigma, which='LM')
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def edge_participation(V: np.ndarray, pos: np.ndarray, R: float, edge_frac: float = 0.2) -> np.ndarray:
    r = np.linalg.norm(pos, axis=1)
    rim = r >= R * (1.0 - edge_frac)
    psi2 = np.abs(V)**2
    num = psi2[rim, :].sum(axis=0)
    den = psi2.sum(axis=0) + 1e-16
    return np.asarray(num / den).ravel()


def pick_edge_state(E: np.ndarray, epr: np.ndarray, target_E: float | None = None) -> int:
    mask = epr > 0.5
    if not np.any(mask):
        return int(np.argmin(np.abs(E)))
    idxs = np.where(mask)[0]
    if target_E is None:
        return int(idxs[np.argmin(np.abs(E[idxs]))])
    return int(idxs[np.argmin(np.abs(E[idxs] - target_E))])


def level_spacing_from_edge_ladder(E: np.ndarray, epr: np.ndarray, epr_thresh: float = 0.5) -> float:
    idx = np.where(epr >= epr_thresh)[0]
    if len(idx) < 3:
        return float('nan')
    Es = np.sort(E[idx])
    d = np.diff(Es)
    return float(np.median(np.abs(d)))


# ----------------------- Plotting helpers -----------------------

def plot_spectrum(E: np.ndarray, epr: np.ndarray, title: str, outpath: str | None = None):
    plt.figure(figsize=(6.4, 4.2))
    x = np.arange(len(E))
    sc = plt.scatter(x, E, c=epr, cmap='plasma', s=10, vmin=0, vmax=1)
    plt.colorbar(sc, label='EPR (edge weight)')
    plt.xlabel('eigenstate index')
    plt.ylabel('energy')
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_ldos(pos: np.ndarray, psi: np.ndarray, R: float, E: float, title_prefix: str, outpath: str | None = None):
    plt.figure(figsize=(5.2, 5.2))
    inten = np.abs(psi)**2
    plt.scatter(pos[:, 0], pos[:, 1], c=inten, s=8, cmap='viridis')
    circ = plt.Circle((0, 0), R, color='k', fill=False, ls='--', lw=1)
    plt.gca().add_patch(circ)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{title_prefix} @ E={E:.4f}")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_spacing_vs_invR(R_list: List[float], Deltas: List[float], v_edge_fit: float | None, outpath: str | None):
    invR = [1.0 / R for R in R_list]
    plt.figure(figsize=(6.0, 4.2))
    plt.scatter(invR, Deltas, s=30)
    if v_edge_fit is not None and np.isfinite(v_edge_fit):
        xs = np.linspace(0, max(invR) * 1.05, 100)
        plt.plot(xs, v_edge_fit * xs, lw=2)
        plt.text(0.05, 0.95, f"v_edge ≈ {v_edge_fit:.3f}", transform=plt.gca().transAxes, va='top')
    plt.xlabel('1/R')
    plt.ylabel('Δ (edge spacing)')
    plt.title('Edge spacing vs 1/R')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_spectral_flow(phi_list: np.ndarray, E_flow: np.ndarray, outpath: str | None = None):
    plt.figure(figsize=(6.6, 4.2))
    for row in E_flow:
        plt.plot(phi_list, row, lw=1.2)
    plt.xlabel('flux Φ/Φ0')
    plt.ylabel('energy')
    plt.title('Spectral flow of low-lying levels')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()


def plot_edge_currents(pos: np.ndarray, H: csr_matrix, psi: np.ndarray,
                       R: float | None = None, edge_frac: float = 0.2,
                       mag_clip: float = 0.95, outpath: str | None = None) -> float:
    rows, cols = H.nonzero()
    xs, ys, us, vs, mags, rim_curr = [], [], [], [], [], []
    rim_start = None
    if R is not None:
        rim_start = R * (1.0 - edge_frac)

    for i, j in zip(rows, cols):
        if i >= j:
            continue
        Hij = H[i, j]
        if Hij == 0:
            continue
        j_ij = -2.0 * np.imag(Hij * np.conj(psi[i]) * psi[j])
        if j_ij == 0:
            continue
        ri, rj = pos[i], pos[j]
        if rim_start is not None and (np.linalg.norm(ri) >= rim_start) and (np.linalg.norm(rj) >= rim_start):
            rim_curr.append(abs(j_ij))
        mid = 0.5 * (ri + rj)
        direction = rj - ri
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        xs.append(mid[0]); ys.append(mid[1])
        us.append(direction[0] * j_ij); vs.append(direction[1] * j_ij)
        mags.append(abs(j_ij))

    mean_rim_current = float(np.mean(rim_curr)) if rim_curr else 0.0
    if not mags:
        return mean_rim_current

    mags = np.array(mags)
    scale = np.quantile(mags, mag_clip)
    us = np.array(us) / (scale + 1e-12)
    vs = np.array(vs) / (scale + 1e-12)

    plt.figure(figsize=(5.8, 5.0))
    plt.scatter(pos[:, 0], pos[:, 1], s=6, c='gray', alpha=0.3)
    plt.quiver(xs, ys, us, vs, angles='xy', scale_units='xy', scale=1, width=0.003)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Edge currents (arrows scaled by magnitude)')
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
    shape: str = "disk"  # {"disk","hex","triangle"}

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
    R_list: List[float] | None = None
    flux_min: float = 0.0
    flux_max: float = 0.5
    flux_points: int = 13

    disorder_W_list: List[float] | None = None
    disorder_seeds: int = 3

    M_min: float = -0.8
    M_max: float = 0.8
    M_points: int = 21


# I/O helpers

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


# Panels

def single_dot_demo(cfg: RunConfig):
    np.random.seed(cfg.seed)
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0)

    h_err = hermiticity_error(H)
    E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
    epr = edge_participation(V, pos, cfg.R, cfg.edge_frac)
    j = pick_edge_state(E, epr)

    plot_spectrum(E, epr, f"Haldane dot R={cfg.R:.1f} (EPR colored)", outpath=os.path.join(cfg.outdir, f"spectrum_R{cfg.R:.1f}.png"))
    plot_ldos(pos, V[:, j], cfg.R, E[j], title_prefix="LDOS (edge state)", outpath=os.path.join(cfg.outdir, f"ldos_R{cfg.R:.1f}.png"))

    mean_rim_current = 0.0
    if cfg.do_currents:
        mean_rim_current = plot_edge_currents(pos, H, V[:, j], R=cfg.R, edge_frac=cfg.edge_frac, outpath=os.path.join(cfg.outdir, "edge_currents.png"))

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
    deltas, records = [], []
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
    plot_spacing_vs_invR(records[:, 0].tolist(), Delta.tolist(), v_edge_fit=v_edge, outpath=os.path.join(cfg.outdir, "spacing_vs_invR.png"))

    save_json({"v_edge": float(v_edge) if np.isfinite(v_edge) else None, "r2": float(r2) if np.isfinite(r2) else None}, os.path.join(cfg.outdir, "metrics_size.json"))

    return deltas, v_edge


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


def disorder_sweep(cfg: RunConfig):
    W_list = cfg.disorder_W_list or [0.0, 0.3, 0.6, 0.9, 1.2]
    rng = np.random.default_rng(cfg.seed)
    records = []
    for W in W_list:
        edge_counts = []
        gaps = []
        for s in range(cfg.disorder_seeds):
            pos, sub, idx = generate_sites(cfg.R, cfg.shape)
            H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0)
            # Add on-site uniform disorder in [-W/2, W/2]
            diag_noise = rng.uniform(-W/2, W/2, size=len(pos))
            H = H.tolil()
            for i, val in enumerate(diag_noise):
                H[i, i] = H[i, i] + val
            H = H.tocsr()
            E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
            epr = edge_participation(V, pos, cfg.R, cfg.edge_frac)
            in_gap = np.where((np.abs(E) < 0.6) & (epr > 0.5))[0]
            edge_counts.append(len(in_gap))
            # crude gap estimate: min positive |E|
            gaps.append(float(np.min(np.abs(E))))
        records.append([W, float(np.mean(edge_counts)), float(np.std(edge_counts)), float(np.mean(gaps))])
        print(f"[disorder] W={W:.2f} mean_edge={np.mean(edge_counts):.2f}")
    records = np.array(records)
    np.savetxt(os.path.join(cfg.outdir, "disorder_sweep.csv"), records, delimiter=",", header="W,edge_count_mean,edge_count_std,gap_est", comments="")

    # small plot
    plt.figure(figsize=(6.2, 4.0))
    plt.errorbar(records[:, 0], records[:, 1], yerr=records[:, 2], fmt='o-', capsize=3)
    plt.xlabel('disorder W')
    plt.ylabel('in-gap edge-state count (mean ± std)')
    plt.title('Edge ladder robustness vs disorder')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, 'disorder_edgecount.png'), dpi=200)
    plt.close()


def phase_sweep(cfg: RunConfig):
    M_list = np.linspace(cfg.M_min, cfg.M_max, cfg.M_points)
    edge_wt, edge_cnt = [], []
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    for M in M_list:
        H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, M, phi_flux=0.0)
        E, V = solve_near_zero(H, k=cfg.k_eigs, sigma=0.0)
        epr = edge_participation(V, pos, cfg.R, cfg.edge_frac)
        edge_wt.append(float(np.mean(epr[np.argsort(np.abs(E))[:20]])))
        edge_cnt.append(int(np.sum((np.abs(E) < 0.6) & (epr > 0.5))))
    edge_wt, edge_cnt = np.array(edge_wt), np.array(edge_cnt)

    Mc = 3 * np.sqrt(3) * cfg.t2 * np.sin(cfg.phi)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(M_list, edge_wt, 'o-', label='mean edge weight (first 20)')
    ax1.set_xlabel('M (Semenoff mass)')
    ax1.set_ylabel('edge weight (EPR)')
    ax2 = ax1.twinx()
    ax2.plot(M_list, edge_cnt, 's--', color='tab:red', label='in-gap edge count')
    ax2.set_ylabel('edge count')
    ax1.axvline(Mc, color='k', ls='--', lw=1, label='M_c')
    ax1.axvline(-Mc, color='k', ls='--', lw=1)
    fig.suptitle('Phase sweep vs M')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, 'phase_sweep_M.png'), dpi=200)
    plt.close(fig)


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

    def slope_stats(E_flow):
        K = min(12, E_flow.shape[0])
        idxs = np.argsort(np.min(np.abs(E_flow), axis=1))[:K]
        dphi = phi_list[-1] - phi_list[0] + 1e-12
        slopes = (E_flow[idxs, -1] - E_flow[idxs, 0]) / dphi
        return float(np.mean(slopes)), float(np.std(slopes))

    m_plus, s_plus = slope_stats(E_plus)
    m_minus, s_minus = slope_stats(E_minus)

    plt.figure(figsize=(6.8, 4.2))
    for row in E_plus:
        plt.plot(phi_list, row, lw=1.0, alpha=0.9, label='+phi' if 'plotted_plus' not in locals() else None)
        plotted_plus = True
    for row in E_minus:
        plt.plot(phi_list, row, lw=1.0, alpha=0.7, linestyle='--', label='-phi' if 'plotted_minus' not in locals() else None)
        plotted_minus = True
    plt.xlabel('flux (Φ/Φ0)')
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
    # Build small dot and export one-particle terms (for JW mapping)
    pos, sub, idx = generate_sites(cfg.R, cfg.shape)
    if len(pos) > cfg.vqe_max_sites:
        # shrink R until under cap
        R_try = cfg.R
        while len(pos) > cfg.vqe_max_sites and R_try > 2.0:
            R_try *= 0.9
            pos, sub, idx = generate_sites(R_try, cfg.shape)
        print(f"[vqe] resized dot to R≈{R_try:.2f} (sites={len(pos)})")
    H = build_haldane_hamiltonian(pos, sub, idx, cfg.t, cfg.t2, cfg.phi, cfg.M, phi_flux=0.0).tocsr()

    rows, cols = H.nonzero()
    terms = []
    for i, j in zip(rows, cols):
        Hij = H[i, j]
        if abs(Hij) == 0:
            continue
        terms.append([int(i), int(j), float(np.real(Hij)), float(np.imag(Hij))])
    terms = np.array(terms)
    np.savetxt(os.path.join(cfg.outdir, 'vqe_terms_ij_ReIm.csv'), terms, delimiter=',', header='i,j,Re,Im', comments='')
    np.savetxt(os.path.join(cfg.outdir, 'vqe_positions_xy.csv'), pos, delimiter=',', header='x,y', comments='')
    np.savetxt(os.path.join(cfg.outdir, 'vqe_sublattice_pm1.csv'), sub, delimiter=',', header='sublattice(+1=A,-1=B)', comments='')

    with open(os.path.join(cfg.outdir, 'vqe_README.txt'), 'w') as f:
        f.write(
            'This folder contains a small Haldane-dot instance exported for VQE scaffolding.\n'
            'Files:\n'
            '  - vqe_terms_ij_ReIm.csv : rows of i,j,Re(H_ij),Im(H_ij).\n'
            '  - vqe_positions_xy.csv  : site coordinates.\n'
            '  - vqe_sublattice_pm1.csv: sublattice labels (+1=A, -1=B).\n'
            '\nUse a fermion-to-qubit mapping (e.g., Jordan-Wigner) to build a qubit Hamiltonian.'
        )
    print('[vqe] Exported scaffold in', cfg.outdir)


# ----------------------- Report bundling (PDF) -----------------------

def _text_block_from_dict(d):
    lines = []
    for k, v in d.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _add_text_page(pdf: PdfPages, title: str, body: str):
    fig = plt.figure(figsize=(8.3, 11.7))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')
    ax.text(0.0, 1.0, title, fontsize=16, weight='bold', va='top')
    ax.text(0.0, 0.95, body, fontsize=10, va='top', family='monospace')
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, img_path: str, title: str | None = None):
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


def bundle_report_pdf(cfg: RunConfig):
    out_pdf = os.path.join(cfg.outdir, cfg.report_name)
    with PdfPages(out_pdf) as pdf:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cover_title = "Haldane Quantum Dot - Benchmark Report"
        body = [
            f"Generated: {now}",
            "",
            "Run configuration:",
            _text_block_from_dict({
                't': cfg.t, 't2': cfg.t2, 'phi': cfg.phi, 'M': cfg.M, 'R': cfg.R,
                'edge_frac': cfg.edge_frac, 'k_eigs': cfg.k_eigs, 'seed': cfg.seed,
                'shape': cfg.shape
            })
        ]
        _add_text_page(pdf, cover_title, "\n".join(body))

        # Metrics pages
        for name, title in [("metrics_single.json", "Single-dot metrics"), ("metrics_size.json", "Size-sweep fit metrics"), ("metrics_phi_flip.json", "Phi-flip slope metrics")]:
            p = os.path.join(cfg.outdir, name)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    m = json.load(f)
                _add_text_page(pdf, title, _text_block_from_dict(m))

        # Figures in order if they exist
        figs = [
            (os.path.join(cfg.outdir, f'spectrum_R{cfg.R:.1f}.png'), 'Spectrum (EPR-colored)'),
            (os.path.join(cfg.outdir, f'ldos_R{cfg.R:.1f}.png'), 'LDOS of representative edge state'),
            (os.path.join(cfg.outdir, 'spacing_vs_invR.png'), 'Edge spacing vs 1/R (fit)'),
            (os.path.join(cfg.outdir, 'spectral_flow.png'), 'Spectral flow under flux'),
            (os.path.join(cfg.outdir, 'edge_currents.png'), 'Edge current map'),
            (os.path.join(cfg.outdir, 'disorder_edgecount.png'), 'Edge ladder robustness vs disorder'),
            (os.path.join(cfg.outdir, 'phase_sweep_M.png'), 'Phase sweep vs M'),
            (os.path.join(cfg.outdir, 'spectral_flow_phi_flip.png'), '+phi vs -phi spectral flow'),
        ]
        for pth, ttl in figs:
            _add_image_page(pdf, pth, ttl)
    print(f"[report] Wrote {out_pdf}")


# ----------------------- CLI -----------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Haldane quantum dot benchmark runner")
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--t2", type=float, default=0.1)
    p.add_argument("--phi", type=float, default=float(np.pi/2))
    p.add_argument("--M", type=float, default=0.2)
    p.add_argument("--R", type=float, default=15.0)
    p.add_argument("--edge_frac", type=float, default=0.2)
    p.add_argument("--k_eigs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="out")

    # geometry
    p.add_argument("--shape", type=str, default="disk", choices=["disk", "hex", "triangle"]) 

    # tasks
    p.add_argument("--do_single", action="store_true")
    p.add_argument("--do_size_sweep", action="store_true")
    p.add_argument("--do_flux", action="store_true")
    p.add_argument("--do_disorder", action="store_true")
    p.add_argument("--do_phase_sweep", action="store_true")
    p.add_argument("--do_currents", action="store_true")

    # report
    p.add_argument("--make_report", action="store_true")
    p.add_argument("--report_name", type=str, default="report.pdf")

    # extras
    p.add_argument("--do_phi_flip_check", action="store_true")
    p.add_argument("--export_vqe_scaffold", action="store_true")
    p.add_argument("--vqe_max_sites", type=int, default=24)

    # size sweep
    p.add_argument("--R_list", type=str, default="")

    # flux
    p.add_argument("--flux_min", type=float, default=0.0)
    p.add_argument("--flux_max", type=float, default=0.5)
    p.add_argument("--flux_points", type=int, default=13)

    # disorder
    p.add_argument("--disorder_W_list", type=str, default="")
    p.add_argument("--disorder_seeds", type=int, default=3)

    # phase
    p.add_argument("--M_min", type=float, default=-0.8)
    p.add_argument("--M_max", type=float, default=0.8)
    p.add_argument("--M_points", type=int, default=21)

    args = p.parse_args()
    cfg = RunConfig()
    cfg.t = args.t; cfg.t2 = args.t2; cfg.phi = args.phi; cfg.M = args.M
    cfg.R = args.R; cfg.edge_frac = args.edge_frac; cfg.k_eigs = args.k_eigs
    cfg.seed = args.seed; cfg.outdir = args.outdir

    cfg.shape = args.shape

    cfg.do_single = args.do_single
    cfg.do_size_sweep = args.do_size_sweep
    cfg.do_flux = args.do_flux
    cfg.do_disorder = args.do_disorder
    cfg.do_phase_sweep = args.do_phase_sweep
    cfg.do_currents = args.do_currents

    cfg.make_report = args.make_report
    cfg.report_name = args.report_name

    cfg.do_phi_flip_check = args.do_phi_flip_check
    cfg.export_vqe_scaffold = args.export_vqe_scaffold
    cfg.vqe_max_sites = args.vqe_max_sites

    if args.R_list:
        cfg.R_list = [float(x) for x in args.R_list.split(',')]
    if args.disorder_W_list:
        cfg.disorder_W_list = [float(x) for x in args.disorder_W_list.split(',')]
    cfg.flux_min = args.flux_min; cfg.flux_max = args.flux_max; cfg.flux_points = args.flux_points
    cfg.M_min = args.M_min; cfg.M_max = args.M_max; cfg.M_points = args.M_points

    ensure_outdir(cfg.outdir)

    # Single-dot panel
    if cfg.do_single:
        single_dot_demo(cfg)

    # Size scaling
    if cfg.do_size_sweep:
        size_sweep(cfg)

    # Flux flow
    if cfg.do_flux:
        flux_flow(cfg)

    # Disorder
    if cfg.do_disorder:
        disorder_sweep(cfg)

    # Phase sweep
    if cfg.do_phase_sweep:
        phase_sweep(cfg)

    # Edge currents
    if cfg.do_currents:
        # already generated during single if requested
        pass

    # phi flip check
    if cfg.do_phi_flip_check:
        phi_flip_check(cfg)

    # VQE export
    if cfg.export_vqe_scaffold:
        export_vqe_scaffold(cfg)

    # Bundle report last
    if cfg.make_report:
        bundle_report_pdf(cfg)


# ==============================================
# File: vqe_run.py  (starter template)
# ==============================================

"""
Usage:
  1) Export a small instance first:
     python haldane_dot_bench_runner.py --export_vqe_scaffold --vqe_max_sites 24 --R 8 --outdir out_vqe
  2) Run this script from the same folder (or set paths below).
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import TwoLocal

# ------------ Load scaffold ------------
TERMS_CSV = 'out_vqe/vqe_terms_ij_ReIm.csv'
POS_CSV   = 'out_vqe/vqe_positions_xy.csv'
SUB_CSV   = 'out_vqe/vqe_sublattice_pm1.csv'

h_ij = np.loadtxt(TERMS_CSV, delimiter=',', skiprows=1)  # i,j,Re,Im
pos  = np.loadtxt(POS_CSV, delimiter=',', skiprows=1)
sub  = np.loadtxt(SUB_CSV, delimiter=',', skiprows=1).astype(int)
N = pos.shape[0]

# Reconstruct single-particle Hamiltonian h
h = np.zeros((N, N), dtype=complex)
for i, j, Re, Im in h_ij:
    i, j = int(i), int(j)
    h[i, j] = Re + 1j*Im
h = (h + h.conj().T)/2

# Decide filling: number of negative eigenvalues at mu=0
vals = np.linalg.eigvalsh(h)
N_occ = int(np.sum(vals < 0.0))

# ------------ Build JW Pauli operator for H ------------
Istr = "I"*N

def add_op(acc: SparsePauliOp, pauli: str, coeff: float):
    op = SparsePauliOp.from_list([(pauli, 1.0)])
    return (acc + coeff*op).simplify()

H_op = SparsePauliOp.from_list([(Istr, 0.0)])

# number operator once (for penalty)
Nhat = SparsePauliOp.from_list([(Istr, 0.0)])
for i in range(N):
    z = list(Istr); z[i] = 'Z'; z = ''.join(z)
    Nhat += 0.5*SparsePauliOp.from_list([(z, -1.0)]) + 0.5*SparsePauliOp.from_list([(Istr, 1.0)])

for i in range(N):
    for j in range(N):
        c = h[i, j]
        Re, Im = float(np.real(c)), float(np.imag(c))
        if abs(Re) < 1e-12 and abs(Im) < 1e-12:
            continue
        if i == j:
            # Re * n_i = Re*(I - Z_i)/2
            z = list(Istr); z[i] = 'Z'; z = ''.join(z)
            H_op = add_op(H_op, z, +Re/2.0)
            H_op = add_op(H_op, Istr, -Re/2.0)
        else:
            lo, hi = (i, j) if i < j else (j, i)
            zstr = list(Istr)
            for k in range(lo+1, hi):
                zstr[k] = 'Z'
            zstr = ''.join(zstr)
            # helpers to place single-qubit ops and merge with Z-string
            def put(op, k):
                s = list(Istr); s[k] = op; return ''.join(s)
            def merge(a, b):
                # overlay a on b where a is not I
                return ''.join(a[q] if a[q] != 'I' else b[q] for q in range(N))
            # Re part: 0.5*(X_i Z... X_j + Y_i Z... Y_j)
            XX = merge(put('X', i), zstr); XX = merge(put('X', j), XX)
            YY = merge(put('Y', i), zstr); YY = merge(put('Y', j), YY)
            H_op = add_op(H_op, XX, +0.5*Re)
            H_op = add_op(H_op, YY, +0.5*Re)
            # Im part: 0.5*i*(X Z... Y - Y Z... X) with ordering sign
            sgn = 1.0 if i < j else -1.0
            XY = merge(put('X', i), zstr); XY = merge(put('Y', j), XY)
            YX = merge(put('Y', i), zstr); YX = merge(put('X', j), YX)
            H_op = add_op(H_op, XY, +0.5*Im*sgn)
            H_op = add_op(H_op, YX, -0.5*Im*sgn)

# ------------ Number penalty to fix particle number ------------
bandwidth = float(np.max(vals) - np.min(vals) + 1e-6)
lam = 10.0 * bandwidth
Iop = SparsePauliOp.from_list([(Istr, 1.0)])
Penalty = (Nhat @ Nhat - 2*N_occ * Nhat + (N_occ**2) * Iop) * lam
H_pen = (H_op + Penalty).simplify()

# ------------ VQE on simulator ------------
algorithm_globals.random_seed = 7
ansatz = TwoLocal(N, rotation_blocks='ry', entanglement_blocks='cz', entanglement='linear', reps=2)
opt = SPSA(maxiter=150)

# Ideal (statevector-like)
estimator_sv = Estimator()
vqe_sv = VQE(estimator_sv, ansatz, opt)
res_sv = vqe_sv.compute_minimum_eigenvalue(operator=H_pen)
print("VQE (statevector) penalized energy:", float(np.real(res_sv.eigenvalue)))

# Shot-based (qasm)
estimator_shot = AerEstimator(shots=2000)
vqe_shot = VQE(estimator_shot, ansatz, SPSA(maxiter=150))
res_shot = vqe_shot.compute_minimum_eigenvalue(operator=H_pen)
print("VQE (shots) penalized energy:", float(np.real(res_shot.eigenvalue)))

# (Optional) from optimal params, measure Z-expectations to build EPR on the rim
# For brevity, measuring EPR is left to a follow-up helper; the idea:
#  - Prepare ansatz(params_opt), measure Z_i for all i (grouped).
#  - Convert to n_i = (1 - Z_i)/2 and compute EPR using the same rim mask as the runner.
