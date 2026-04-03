#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.sparse.linalg import LinearOperator, eigsh
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix, identity
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import multiprocessing as mp
    MP_OK = True
except Exception:
    MP_OK = False

# ----------------------------- Model -----------------------------

class HaldaneRealSpace:
    """
    Real-space Haldane model on a finite honeycomb (open boundaries).
    - NN hopping:        -t1
    - NNN hopping:       -t2 * exp(i * nu_ij * phi), nu_ij = ±1 (chirality)
    - Sublattice mass:   +M on A, -M on B
    Geometry: Lx x Ly unit cells, 2 sites per cell (N = 2*Lx*Ly).
    """
    def __init__(self, Lx=10, Ly=6, t1=1.0, t2=0.1, phi=np.pi/2, M=0.0, seed=None):
        if not SCIPY_OK:
            raise RuntimeError("SciPy is required for large-scale runs. Install with: pip install scipy")
        self.Lx = int(Lx); self.Ly = int(Ly)
        self.t1 = float(t1); self.t2 = float(t2)
        self.phi = float(phi); self.M = float(M)
        self.rng = np.random.default_rng(seed)

        # Geometry (units so NN distance = 1)
        self.d1 = np.array([0.0, -1.0])
        self.d2 = np.array([np.sqrt(3)/2, 0.5])
        self.d3 = np.array([-np.sqrt(3)/2, 0.5])
        self.a1 = self.d2 - self.d3          # (sqrt(3), 0)
        self.a2 = self.d3 - self.d1          # (-sqrt(3)/2, 3/2)
        self.rA = np.array([0.0, 0.0])
        self.rB = self.d1.copy()

        self.r_nn = 1.0
        self.r_nnn = np.linalg.norm(self.a1)  # sqrt(3)

        self.pos = None
        self.subl = None
        self.N = None
        self.nn_list = None  # adjacency sets (for chirality)

    # ---------- lattice and neighbors ----------

    def _build_lattice(self):
        positions, subl = [], []
        for y in range(self.Ly):
            for x in range(self.Lx):
                R = x * self.a1 + y * self.a2
                positions.append(R + self.rA); subl.append(+1)  # A
                positions.append(R + self.rB); subl.append(-1)  # B
        self.pos  = np.array(positions, dtype=np.float64)
        self.subl = np.array(subl, dtype=np.int8)
        self.N    = self.pos.shape[0]

    def _find_neighbors(self):
        tree = cKDTree(self.pos)
        nn_pairs, nnn_pairs = [], []
        eps = 1e-3
        for i in range(self.N):
            cand = tree.query_ball_point(self.pos[i], r=self.r_nn + eps)
            for j in cand:
                if j <= i or self.subl[i] == self.subl[j]:
                    continue
                if abs(np.linalg.norm(self.pos[j]-self.pos[i]) - self.r_nn) < 1e-3:
                    nn_pairs.append((i, j))
            cand2 = tree.query_ball_point(self.pos[i], r=self.r_nnn + eps)
            for j in cand2:
                if j <= i or self.subl[i] != self.subl[j]:
                    continue
                if abs(np.linalg.norm(self.pos[j]-self.pos[i]) - self.r_nnn) < 1e-3:
                    nnn_pairs.append((i, j))
        nn_adj = [set() for _ in range(self.N)]
        for i, j in nn_pairs:
            nn_adj[i].add(j); nn_adj[j].add(i)
        self.nn_list = nn_adj
        return nn_pairs, nnn_pairs

    def _chirality_nu(self, i, j):
        common = self.nn_list[i].intersection(self.nn_list[j])
        if not common:
            return 0
        k = next(iter(common))
        v1 = self.pos[k] - self.pos[i]
        v2 = self.pos[j] - self.pos[k]
        cross_z = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross_z) < 1e-12 and len(common) > 1:
            k2 = list(common)[1]
            v1 = self.pos[k2] - self.pos[i]
            v2 = self.pos[j] - self.pos[k2]
            cross_z = v1[0]*v2[1] - v1[1]*v2[0]
        return 1 if cross_z > 0 else -1

    # ---------- Hamiltonian ----------

    def hamiltonian(self):
        self._build_lattice()
        nn_pairs, nnn_pairs = self._find_neighbors()

        rows, cols, vals = [], [], []
        # On-site mass
        for i in range(self.N):
            rows.append(i); cols.append(i); vals.append(self.M * float(self.subl[i]))
        # NN
        for i, j in nn_pairs:
            rows += [i, j]; cols += [j, i]; vals += [-self.t1, -self.t1]
        # NNN with complex phase
        cphi, sphi = math.cos(self.phi), math.sin(self.phi)
        for i, j in nnn_pairs:
            nu = self._chirality_nu(i, j)
            hij = -self.t2 * complex(cphi, nu * sphi)
            rows += [i, j]; cols += [j, i]; vals += [hij, np.conjugate(hij)]
        H = sp.coo_matrix((np.array(vals, dtype=np.complex128),
                           (np.array(rows), np.array(cols))),
                          shape=(self.N, self.N)).tocsr()
        # enforce Hermiticity in case of tiny numeric asymmetries
        H = (H + H.getH()) * 0.5
        return H

    # ---------- eigen solve near zero (fast) ----------

    def eigsolve_near_zero(self, k=16, sigma=0.0, maxiter=None, tol=1e-8, use_shift_invert=True):
        """
        Compute k eigenpairs closest to sigma (default 0).
        If use_shift_invert: pre-factorize H - sigma*I to accelerate.
        """
        H = self.hamiltonian()
        t0 = time.perf_counter()
        if use_shift_invert:
            A = (H - sigma * identity(H.shape[0], dtype=H.dtype, format="csr"))
            lu = spla.splu(A.tocsc(), permc_spec="COLAMD")  # sparse LU (SuperLU)
            def op(x):
                return lu.solve(x)
            OPinv = LinearOperator(shape=H.shape, matvec=op, dtype=H.dtype)
            # ARPACK on OPinv*A ≈ identity shifts spectrum around sigma
            vals, vecs = eigsh(H, k=min(k, H.shape[0]-2), sigma=sigma, which="LM",
                               OPinv=OPinv, maxiter=maxiter, tol=tol)
        else:
            vals, vecs = eigsh(H, k=min(k, H.shape[0]-2), which="SM", maxiter=maxiter, tol=tol)
        vals = np.real(vals)
        idx = np.argsort(np.abs(vals - sigma))
        vals, vecs = vals[idx], vecs[:, idx]
        t1 = time.perf_counter()
        return vals, vecs, (t1 - t0)

    # ---------- diagnostics ----------

    def edge_localization(self, vec, margin=None):
        if margin is None:
            margin = 1.5 * self.r_nn
        x = self.pos[:, 0]; y = self.pos[:, 1]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        edge_mask = ((x - xmin < margin) | (xmax - x < margin) |
                     (y - ymin < margin) | (ymax - y < margin))
        prob = np.abs(vec)**2
        return float(prob[edge_mask].sum() / prob.sum())

# ----------------------------- Plotting -----------------------------

def plot_spectrum(evals, outfile="spectrum_near_zero.png"):
    fig = plt.figure(figsize=(5, 3.2))
    x = np.arange(len(evals))
    plt.scatter(x, evals, s=16)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Mode index (sorted near 0)")
    plt.ylabel("Energy")
    plt.title("Eigenvalues near zero")
    plt.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)

def plot_edge_state(model, vec, outfile="edge_state_density.png"):
    prob = np.abs(vec)**2
    fig = plt.figure(figsize=(4.6, 4.2))
    plt.scatter(model.pos[:, 0], model.pos[:, 1], c=prob, s=10)
    plt.axis("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("|ψ|² (near‑zero mode)")
    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_scaling(stats, prefix="scaling"):
    N = np.array([row["N_sites"] for row in stats], dtype=float)
    nnz = np.array([row["nnz"] for row in stats], dtype=float)
    build_s = np.array([row["build_s"] for row in stats], dtype=float)
    eig_s = np.array([row["eig_s"] for row in stats], dtype=float)

    # Eigsolve time vs N
    fig1 = plt.figure(figsize=(5, 3.2))
    plt.plot(N, eig_s, marker="o")
    plt.xlabel("Number of sites N")
    plt.ylabel("Eigsolve time (s)")
    plt.title("Eigsolve time vs size")
    plt.tight_layout()
    fig1.savefig(f"{prefix}_eig_time.png", dpi=180); plt.close(fig1)

    # Build time vs N
    fig2 = plt.figure(figsize=(5, 3.2))
    plt.plot(N, build_s, marker="o")
    plt.xlabel("Number of sites N")
    plt.ylabel("Build time (s)")
    plt.title("Hamiltonian build vs size")
    plt.tight_layout()
    fig2.savefig(f"{prefix}_build_time.png", dpi=180); plt.close(fig2)

    # Nonzeros vs N
    fig3 = plt.figure(figsize=(5, 3.2))
    plt.plot(N, nnz, marker="o")
    plt.xlabel("Number of sites N")
    plt.ylabel("Nonzeros (nnz)")
    plt.title("Matrix nonzeros vs size")
    plt.tight_layout()
    fig3.savefig(f"{prefix}_nnz.png", dpi=180); plt.close(fig3)

    # Density vs N
    fig4 = plt.figure(figsize=(5, 3.2))
    plt.plot(N, nnz / (N * N), marker="o")
    plt.xlabel("Number of sites N")
    plt.ylabel("Density nnz / N²")
    plt.title("Sparsity density vs size")
    plt.tight_layout()
    fig4.savefig(f"{prefix}_density.png", dpi=180); plt.close(fig4)

# ----------------------------- Scaling runner -----------------------------

def run_one_size(args_tuple):
    (lx, ly, t1, t2, phi, M, k, sigma, maxiter, tol, use_shift_invert) = args_tuple
    model = HaldaneRealSpace(Lx=lx, Ly=ly, t1=t1, t2=t2, phi=phi, M=M)
    t0 = time.perf_counter()
    H = model.hamiltonian()
    t1b = time.perf_counter()
    build_time = t1b - t0
    nnz = int(H.nnz)
    w, v, t_eig = model.eigsolve_near_zero(k=k, sigma=sigma, maxiter=maxiter,
                                           tol=tol, use_shift_invert=use_shift_invert)
    return {
        "Lx": lx, "Ly": ly, "N_sites": model.N, "nnz": nnz,
        "build_s": build_time, "eig_s": t_eig,
        "min_abs_E": float(np.min(np.abs(w))) if len(w) else np.nan
    }

def scaling_study_parallel(sizes, t1, t2, phi, M, k=8, sigma=0.0, maxiter=None,
                           tol=1e-8, use_shift_invert=True, nproc=None):
    args = [(lx, ly, t1, t2, phi, M, k, sigma, maxiter, tol, use_shift_invert) for (lx, ly) in sizes]
    if MP_OK:
        with mp.Pool(processes=nproc) as pool:
            rows = pool.map(run_one_size, args)
    else:
        rows = [run_one_size(a) for a in args]
    # ensure increasing by N
    rows = sorted(rows, key=lambda r: r["N_sites"])
    return rows

def save_csv(stats, path="scaling_stats.csv"):
    if not stats:
        return
    keys = list(stats[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in stats:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

# ----------------------------- CLI -----------------------------

def parse_sizes(s):
    out = []
    for item in s.split(","):
        x, y = item.lower().split("x")
        out.append((int(x), int(y)))
    return out

def main():
    p = argparse.ArgumentParser(description="Large-scale real-space Haldane simulator with scaling plots.")
    p.add_argument("--Lx", type=int, default=16, help="Unit cells in x (for single-run visualization)")
    p.add_argument("--Ly", type=int, default=10, help="Unit cells in y (for single-run visualization)")
    p.add_argument("--t1", type=float, default=1.0)
    p.add_argument("--t2", type=float, default=0.1)
    p.add_argument("--phi", type=float, default=np.pi/2)
    p.add_argument("--M", type=float, default=0.0)
    p.add_argument("--k", type=int, default=16, help="Eigenpairs near sigma to compute")
    p.add_argument("--sigma", type=float, default=0.0, help="Target energy for shift-invert")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--no_shift_invert", action="store_true", help="Disable shift-invert (slower for large N)")
    p.add_argument("--sizes", type=str, default="12x12,20x12,32x20,48x32,64x40",
                   help="Comma-separated list, e.g. 12x12,20x12,32x20")
    p.add_argument("--nproc", type=int, default=None, help="Processes for parallel scaling study")
    p.add_argument("--no_plots", action="store_true", help="Skip all plotting (useful for speed testing)")
    p.add_argument("--prefix", type=str, default="scaling", help="Prefix for scaling plot files")
    p.add_argument("--csv", type=str, default="scaling_stats.csv", help="CSV path for scaling results")
    args = p.parse_args()

    use_shift_invert = not args.no_shift_invert

    # 1) Single visualization run (spectrum + edge mode)
    model = HaldaneRealSpace(Lx=args.Lx, Ly=args.Ly, t1=args.t1, t2=args.t2, phi=args.phi, M=args.M)
    w, V, t_eig = model.eigsolve_near_zero(k=args.k, sigma=args.sigma, tol=args.tol,
                                           maxiter=args.maxiter, use_shift_invert=use_shift_invert)
    print(f"[viz {args.Lx}x{args.Ly}] N_sites={model.N}, eig_time={t_eig:.3f}s")
    print("Eigenvalues near sigma:")
    print(np.array2string(w, precision=6, suppress_small=True))

    if not args.no_plots:
        plot_spectrum(w, outfile="spectrum_near_zero.png")
        if V.shape[1] >= 1:
            edge_frac = model.edge_localization(V[:, 0])
            print(f"Edge fraction (lowest |E| mode): {edge_frac:.3f}")
            plot_edge_state(model, V[:, 0], outfile="edge_state_density.png")

    # 2) Parallel scaling study
    sizes = parse_sizes(args.sizes)
    t0 = time.perf_counter()
    stats = scaling_study_parallel(
        sizes=sizes, t1=args.t1, t2=args.t2, phi=args.phi, M=args.M,
        k=min(args.k, 12),  # keep k modest for faster scaling sweeps
        sigma=args.sigma, maxiter=args.maxiter, tol=args.tol,
        use_shift_invert=use_shift_invert, nproc=args.nproc
    )
    t1b = time.perf_counter()
    print("\nScaling summary:")
    for row in stats:
        pretty = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
        print(pretty)
    print(f"\nScaling wall time: {t1b - t0:.3f}s with sizes: {sizes}")

    # Save CSV + plots
    save_csv(stats, path=args.csv)
    if not args.no_plots:
        plot_scaling(stats, prefix=args.prefix)

    if not args.no_plots:
        print("\nSaved figures:")
        print("- spectrum_near_zero.png")
        print("- edge_state_density.png")
        print(f"- {args.prefix}_eig_time.png")
        print(f"- {args.prefix}_build_time.png")
        print(f"- {args.prefix}_nnz.png")
        print(f"- {args.prefix}_density.png")
    print(f"Saved CSV: {args.csv}")

if __name__ == "__main__":
    main()
