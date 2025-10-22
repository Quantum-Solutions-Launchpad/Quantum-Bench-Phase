#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from dataclasses import dataclass, asdict
import csv
from typing import Tuple, Dict, List

## ------------------------------ Utilities ------------------------------ ##
def ensure_dir(path: str):
    ## create directory if missing ##
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed: int):
    ## set numpy rng ##
    if seed is not None:
        np.random.seed(seed)

## -------------------------- Model + Observables ------------------------- ##
@dataclass
class Params:
    N: int = 200                ## number of sites ##
    t: float = 1.0              ## hopping amplitude ##
    Delta: float = 0.2          ## p-wave pairing ##
    mu: float = 0.0             ## mean onsite chemical potential ##
    W: float = 0.0              ## disorder strength for mu_i ~ N(mu, W^2) ##
    soft_boundary: bool = False ## apply soft confinement profile on mu ##
    V0: float = 0.0             ## strength of soft boundary offset near edges ##
    xi: float = 1.0             ## decay length (in sites) for soft boundary ##
    realizations: int = 50      ## number of disorder draws ##
    seed: int = 0               ## rng seed ##
    edge_window: int = 8        ## sites considered an 'edge' for participation ##
    outdir: str = "out_mrb"     ## output folder ##
    tag: str = "kitaev"         ## label for outputs ##

def majorana_phase(mu: float, t: float) -> int:
    ## Kitaev-chain bulk invariant: topological if |mu| < 2t ##
    return -1 if abs(mu) < 2.0 * abs(t) else +1

def soft_profile(N: int, V0: float, xi: float) -> np.ndarray:
    ## exponential soft walls at both ends added to mu ##
    x = np.arange(N)
    left = V0 * np.exp(-x / xi)
    right = V0 * np.exp(-(N - 1 - x) / xi)
    return left + right

def draw_mu_vector(p: Params) -> np.ndarray:
    ## sample mu_i, optionally add soft boundary ##
    mu_vec = np.random.normal(loc=p.mu, scale=p.W, size=p.N) if p.W > 0 else np.full(p.N, p.mu)
    if p.soft_boundary and p.V0 != 0.0:
        mu_vec = mu_vec + soft_profile(p.N, p.V0, p.xi)
    return mu_vec

def bdg_matrix_open_chain(mu_vec: np.ndarray, t: float, Delta: float) -> np.ndarray:
    ## construct 2N x 2N BdG Hamiltonian in Nambu basis (c, c^\dagger) with open BC ##
    ## H = [[H0,  Delta_mat],[Delta_mat^T, -H0^T]] for spinless p-wave ##
    N = len(mu_vec)
    H0 = np.zeros((N, N), dtype=np.float64)
    ## onsite ##
    np.fill_diagonal(H0, -mu_vec)
    ## nearest neighbor hopping ##
    for i in range(N-1):
        H0[i, i+1] = -t
        H0[i+1, i] = -t
    ## pairing (p-wave between neighbors) ##
    Delta_mat = np.zeros((N, N), dtype=np.float64)
    for i in range(N-1):
        Delta_mat[i, i+1] = +Delta
        Delta_mat[i+1, i] = -Delta  ## antisymmetric ##
    H_bdg = np.block([[H0,        Delta_mat],
                      [Delta_mat.T, -H0.T   ]])
    return H_bdg

def edge_participation(psi: np.ndarray, edge_window: int, N: int) -> Tuple[float, float]:
    ## psi is length 2N eigvec (u,v). Weight per site = |u_j|^2 + |v_j|^2 ##
    u = psi[:N]
    v = psi[N:]
    w_site = np.abs(u)**2 + np.abs(v)**2
    left = np.sum(w_site[:edge_window])
    right = np.sum(w_site[-edge_window:])
    return float(left), float(right)

def inverse_participation_ratio(psi: np.ndarray, N: int) -> float:
    ## IPR over sites using total weight per site ##
    u = psi[:N]
    v = psi[N:]
    w_site = np.abs(u)**2 + np.abs(v)**2
    w_site = w_site / np.sum(w_site)
    return float(np.sum(w_site**2))

def localization_length(psi: np.ndarray, N: int) -> float:
    ## fit log envelope to estimate xi_loc; robust to noise using median-based slope ##
    u = psi[:N]
    v = psi[N:]
    w = np.abs(u)**2 + np.abs(v)**2
    w = w / (np.sum(w) + 1e-16)
    x = np.arange(N)
    ## avoid zeros ##
    mask = w > (1e-12 * np.max(w))
    if np.sum(mask) < 5:
        return np.inf
    xs = x[mask]
    ys = np.log(w[mask])
    ## linear fit via least squares ##
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    ## if peak is at an edge, slope ~ -1/xi; take abs ##
    xi_est = 1.0 / max(1e-12, abs(m))
    return float(xi_est)

def find_zero_pair(evals: np.ndarray, evecs: np.ndarray) -> Tuple[float, np.ndarray]:
    ## return |E_min| and eigenvector associated with closest-to-zero energy ##
    idx = np.argmin(np.abs(evals))
    return float(evals[idx]), evecs[:, idx]

## -------------------------------- Runner -------------------------------- ##
def run_once(p: Params, mu_vec: np.ndarray) -> Dict[str, float]:
    H = bdg_matrix_open_chain(mu_vec, p.t, p.Delta)
    evals, evecs = eigh(H)  ## BdG is hermitian ##
    Emin, psi0 = find_zero_pair(evals, evecs)
    N = p.N

    left_w, right_w = edge_participation(psi0, p.edge_window, N)
    edge_overlap = 2.0 * min(left_w, right_w)  ## 0..1, best Majoranas ~ both edges ##
    ipr = inverse_participation_ratio(psi0, N)
    xi_loc = localization_length(psi0, N)

    return {
        "Emin": abs(Emin),
        "left_edge_weight": left_w,
        "right_edge_weight": right_w,
        "edge_overlap": edge_overlap,
        "ipr": ipr,
        "xi_loc": xi_loc
    }

def sweep(p: Params) -> Dict[str, float]:
    set_seed(p.seed)
    topo = majorana_phase(p.mu, p.t)

    rows = []
    for r in range(p.realizations):
        mu_vec = draw_mu_vector(p)
        res = run_once(p, mu_vec)
        res.update({"realization": r})
        rows.append(res)

    ## aggregate ##
    agg = {k: float(np.mean([row[k] for row in rows])) for k in rows[0].keys() if k != "realization"}
    agg_std = {k+"_std": float(np.std([row[k] for row in rows])) for k in rows[0].keys() if k != "realization"}

    out = {"topo_invariant": topo, **agg, **agg_std}
    return out, rows

## --------------------------------- I/O ---------------------------------- ##
def save_csv(rows: List[Dict[str, float]], path: str):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def plot_hist(rows: List[Dict[str, float]], key: str, title: str, outpng: str):
    vals = np.array([r[key] for r in rows])
    plt.figure()
    plt.hist(vals, bins=40)
    plt.title(title)
    plt.xlabel(key)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def plot_profile_example(p: Params, mu_vec: np.ndarray, outpng: str):
    plt.figure()
    plt.plot(mu_vec, lw=2)
    plt.title("On-site chemical potential profile (example realization)")
    plt.xlabel("site index")
    plt.ylabel(r"$\mu_i$")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def quick_diagnostics(p: Params, rows: List[Dict[str, float]], outdir: str):
    plot_hist(rows, "Emin", "Lowest |E| across realizations", os.path.join(outdir, f"{p.tag}_hist_Emin.png"))
    plot_hist(rows, "edge_overlap", "Edge overlap across realizations", os.path.join(outdir, f"{p.tag}_hist_edgeoverlap.png"))
    plot_hist(rows, "ipr", "IPR across realizations", os.path.join(outdir, f"{p.tag}_hist_ipr.png"))
    plot_hist(rows, "xi_loc", "Localization length across realizations", os.path.join(outdir, f"{p.tag}_hist_xiloc.png"))

## ------------------------------ CLI / Main ----------------------------- ##
def main():
    ap = argparse.ArgumentParser(description="Majorana Robustness Benchmark (Kitaev chain, open BC)")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--Delta", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--W", type=float, default=0.0)
    ap.add_argument("--soft_boundary", action="store_true")
    ap.add_argument("--V0", type=float, default=0.0)
    ap.add_argument("--xi", type=float, default=1.5)
    ap.add_argument("--realizations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edge_window", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="out_mrb")
    ap.add_argument("--tag", type=str, default="kitaev")
    args = ap.parse_args()

    p = Params(**vars(args))
    ensure_dir(p.outdir)

    ## single illustrative mu profile ##
    set_seed(p.seed)
    mu_vec_example = draw_mu_vector(p)
    plot_profile_example(p, mu_vec_example, os.path.join(p.outdir, f"{p.tag}_mu_profile.png"))

    ## full sweep ##
    out_agg, rows = sweep(p)

    ## save rows + summary ##
    save_csv(rows, os.path.join(p.outdir, f"{p.tag}_samples.csv"))
    with open(os.path.join(p.outdir, f"{p.tag}_summary.json"), "w") as f:
        json.dump(out_agg, f, indent=2)

    ## plots ##
    quick_diagnostics(p, rows, p.outdir)

    ## stdout summary ##
    print(json.dumps(out_agg, indent=2))
    print(f"Saved per-realization metrics to: {os.path.join(p.outdir, f'{p.tag}_samples.csv')}")
    print(f"Saved summary to: {os.path.join(p.outdir, f'{p.tag}_summary.json')}")
    print(f"Saved histograms to: {p.outdir}")

if __name__ == "__main__":
    main()
