#!/usr/bin/env python3

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

## ---------------------------- helpers / profiles ---------------------------- ##
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def soft_profile(N, V0, xi):
    x = np.arange(N)
    left = V0 * np.exp(-x / xi)
    right = V0 * np.exp(-(N - 1 - x) / xi)
    return left + right

def draw_mu_vec(N, mu, W, soft_on=False, V0=0.0, xi=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mu_vec = np.random.normal(mu, W, size=N) if W > 0 else np.full(N, mu)
    if soft_on and V0 != 0.0:
        mu_vec = mu_vec + soft_profile(N, V0, xi)
    return mu_vec

## ------------------------ spinful nanowire BdG builder ---------------------- ##
## basis per site j: [c_{j,↑}, c_{j,↓}, c_{j,↑}^\dagger, c_{j,↓}^\dagger] so total dim = 4N ##
## parameters: t (hop), alpha (Rashba), mu_j (onsite), Vz (Zeeman x), Delta (s-wave) ##
def bdg_nanowire(mu_vec, t, alpha, Vz, Delta):
    N = len(mu_vec)
    dim = 4 * N
    H = np.zeros((dim, dim), dtype=np.complex128)

    ## on-site terms ##
    for j in range(N):
        idx = 4 * j
        ## electron block He: (-mu_j) I_spin + Vz * sigma_x ##
        He = np.array([[-mu_vec[j], Vz],
                       [Vz,        -mu_vec[j]]], dtype=np.complex128)
        ## hole block Hh: +mu_j I_spin - Vz * sigma_x (i.e., -He^T with sign flips from PH symmetry) ##
        Hh = -He.conj()
        ## pairing Delta i sigma_y in e-h off-diagonals ##
        D = 1j * Delta * np.array([[0, 1], [-1, 0]], dtype=np.complex128)

        ## place onsite blocks ##
        H[idx:idx+2, idx:idx+2] += He
        H[idx+2:idx+4, idx+2:idx+4] += Hh
        H[idx:idx+2, idx+2:idx+4] += D
        H[idx+2:idx+4, idx:idx+2] += D.conj().T

    ## nearest-neighbor hopping and Rashba ##
    for j in range(N - 1):
        a = 4 * j
        b = 4 * (j + 1)

        ## electron block neighbor coupling: -t I + i alpha sigma_y ##
        hop_e = -t * np.eye(2, dtype=np.complex128) + 1j * alpha * np.array([[0, -1], [1, 0]], dtype=np.complex128)
        ## hole block: conjugate with opposite sign -> +t I - i alpha sigma_y ##
        hop_h = hop_e.conj()

        ## place e-e and h-h couplings (both directions) ##
        H[a:a+2, b:b+2] += hop_e
        H[b:b+2, a:a+2] += hop_e.conj().T
        H[a+2:a+4, b+2:b+4] += hop_h
        H[b+2:b+4, a+2:a+4] += hop_h.conj().T

    return H

## ------------------------------ observables -------------------------------- ##
def edge_metrics(psi, N, edge_window=8):
    ## split u (electrons) and v (holes), each has spin up/down over N sites ##
    u = psi[:2*N]
    v = psi[2*N:]
    ## site-resolved weight: sum over spin channels ##
    w_site = np.zeros(N, dtype=float)
    for j in range(N):
        u_up = u[2*j+0]; u_dn = u[2*j+1]
        v_up = v[2*j+0]; v_dn = v[2*j+1]
        w_site[j] = (abs(u_up)**2 + abs(u_dn)**2 + abs(v_up)**2 + abs(v_dn)**2).real
    w_site = w_site / (np.sum(w_site) + 1e-16)
    left = float(np.sum(w_site[:edge_window]))
    right = float(np.sum(w_site[-edge_window:]))
    edge_overlap = 2.0 * min(left, right)
    ipr = float(np.sum(w_site**2))
    ## electron-hole balance: 1 - |sum(|u|^2 - |v|^2)| ##
    eu = float(np.sum(np.abs(u)**2))
    ev = float(np.sum(np.abs(v)**2))
    eh_balance = 1.0 - abs((eu - ev) / (eu + ev + 1e-16))
    return left, right, edge_overlap, ipr, eh_balance

def run_once(N, t, alpha, Vz, Delta, mu, W, soft_boundary, V0, xi, edge_window, seed=None):
    mu_vec = draw_mu_vec(N, mu, W, soft_boundary, V0, xi, seed)
    H = bdg_nanowire(mu_vec, t, alpha, Vz, Delta)
    evals, evecs = eigh(H)
    idx = np.argsort(np.abs(evals))[:2]
    Es = evals[idx]
    Psis = evecs[:, idx]
    ## combine the doublet density as a subspace (reduces edge-selection artifacts) ##
    psi_sub = Psis[:, 0] / np.linalg.norm(Psis[:, 0] + 1e-16) + Psis[:, 1] / np.linalg.norm(Psis[:, 1] + 1e-16)
    left, right, edge_overlap, ipr, ehb = edge_metrics(psi_sub, N, edge_window=edge_window)
    return {
        "Emin": float(np.min(np.abs(Es))),
        "left_edge_weight": left,
        "right_edge_weight": right,
        "edge_overlap": edge_overlap,
        "ipr": ipr,
        "eh_balance": ehb
    }

## ---------------------------- phase diagram sweep --------------------------- ##
def sweep_phase(N, t, alpha, Delta, mu_list, Vz_list, W, soft_boundary, V0, xi, edge_window, seed=None):
    grid = np.zeros((len(Vz_list), len(mu_list)), dtype=float)
    for i, Vz in enumerate(Vz_list):
        for j, mu in enumerate(mu_list):
            res = run_once(N, t, alpha, Vz, Delta, mu, W, soft_boundary, V0, xi, edge_window, seed)
            grid[i, j] = res["Emin"]
    return grid

def plot_phase(mu_list, Vz_list, grid, outpng, title):
    plt.figure(figsize=(6.2,4.8))
    im = plt.imshow(grid, origin="lower", aspect="auto",
                    extent=[min(mu_list), max(mu_list), min(Vz_list), max(Vz_list)],
                    cmap="viridis")
    plt.colorbar(im, label="min |E|")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$V_Z$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()

## --------------------------------- CLI ------------------------------------- ##
def main():
    ap = argparse.ArgumentParser(description="Spinful Rashba nanowire Majorana benchmark")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--Delta", type=float, default=0.25)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--Vz", type=float, default=0.8)
    ap.add_argument("--W", type=float, default=0.0)
    ap.add_argument("--soft_boundary", action="store_true")
    ap.add_argument("--V0", type=float, default=0.0)
    ap.add_argument("--xi", type=float, default=6.0)
    ap.add_argument("--edge_window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="out_nanowire")
    ap.add_argument("--tag", type=str, default="wire")

    ap.add_argument("--do_phase", action="store_true")
    ap.add_argument("--mu_min", type=float, default=-3.0)
    ap.add_argument("--mu_max", type=float, default=3.0)
    ap.add_argument("--mu_pts", type=int, default=61)
    ap.add_argument("--Vz_min", type=float, default=0.0)
    ap.add_argument("--Vz_max", type=float, default=2.5)
    ap.add_argument("--Vz_pts", type=int, default=61)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    ## single instance ##
    res = run_once(args.N, args.t, args.alpha, args.Vz, args.Delta, args.mu,
                   args.W, args.soft_boundary, args.V0, args.xi, args.edge_window, seed=args.seed)
    with open(os.path.join(args.outdir, f"{args.tag}_single.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    ## optional phase diagram ##
    if args.do_phase:
        mu_list = np.linspace(args.mu_min, args.mu_max, args.mu_pts)
        Vz_list = np.linspace(args.Vz_min, args.Vz_max, args.Vz_pts)
        grid = sweep_phase(args.N, args.t, args.alpha, args.Delta, mu_list, Vz_list,
                           args.W, args.soft_boundary, args.V0, args.xi, args.edge_window, seed=args.seed)
        np.save(os.path.join(args.outdir, f"{args.tag}_grid.npy"), grid)
        plot_phase(mu_list, Vz_list, grid,
                   os.path.join(args.outdir, f"{args.tag}_phase.png"),
                   f"Nanowire min |E| heatmap (N={args.N}, Δ={args.Delta}, α={args.alpha})")

if __name__ == "__main__":
    main()
