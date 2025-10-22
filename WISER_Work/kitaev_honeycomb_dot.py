#!/usr/bin/env python3

import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

## lattice vectors (triangular Bravais) ##
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])

## sublattice positions within a cell (for pretty plotting) ##
## choose B displaced by a small vector so A,B are distinct visually ##
rA = np.array([0.0, 0.0])
rB = np.array([0.0, 1.0/np.sqrt(3)])

## NN connectivity encoded by which B cell an A connects to for x,y,z bonds ##
## convention mirrors the k-space model used earlier ##
## A(n,m) connects to:
##  - z: B(n,m)
##  - x: B(n-1,m)
##  - y: B(n,m-1)
NN_SHIFTS = {
    "z": (0, 0),
    "x": (-1, 0),
    "y": (0, -1),
}

## NNN connectivity (same sublattice) along triangular vectors with Haldane signs ##
## use three oriented steps: +a1, +a2, +(a2 - a1) ##
## implement as cell shifts for A sites; B gets opposite sign convention ##
NNN_STEPS = [
    (+1, 0),  ## +a1
    (0, +1),  ## +a2
    (-1, +1), ## +a2 - a1
]

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def honeycomb_positions(Lx, Ly):
    ## build positions of unit-cell origins in Cartesian ##
    cells = []
    for n in range(-Lx, Lx+1):
        for m in range(-Ly, Ly+1):
            R = n*a1 + m*a2
            cells.append((n, m, R))
    return cells

def build_island_indices(R_radius, Lx, Ly):
    ## pick cells within a circle of radius R_radius (in units of |a1|) ##
    cells = honeycomb_positions(Lx, Ly)
    A_sites = []
    B_sites = []
    for n, m, R in cells:
        RA = R + rA
        RB = R + rB
        if np.linalg.norm(RA) <= R_radius:
            A_sites.append((n, m))
        if np.linalg.norm(RB) <= R_radius:
            B_sites.append((n, m))
    ## index map ##
    idxA = { (n,m): i for i,(n,m) in enumerate(sorted(A_sites)) }
    idxB = { (n,m): i for i,(n,m) in enumerate(sorted(B_sites)) }
    return idxA, idxB

def site_position(subl, n, m):
    R = n*a1 + m*a2
    return R + (rA if subl=="A" else rB)

def build_hamiltonian(Kx, Ky, Kz, kappa, R_radius, Lx, Ly):
    idxA, idxB = build_island_indices(R_radius, Lx, Ly)
    NA = len(idxA); NB = len(idxB)
    N = NA + NB
    H = np.zeros((N, N), dtype=np.complex128)

    def add_AB(A_nm, B_nm, K):
        if A_nm in idxA and B_nm in idxB:
            i = idxA[A_nm]
            j = NB + idxB[B_nm]
            H[i, j] += K
            H[j, i] += np.conj(K)

    def add_NNN(subl, nm1, nm2, sign):
        if subl == "A":
            if nm1 in idxA and nm2 in idxA:
                i = idxA[nm1]; j = idxA[nm2]
                val = 1j * kappa * sign
                H[i, j] += val
                H[j, i] += -np.conj(val)
        else:
            if nm1 in idxB and nm2 in idxB:
                i = NB + idxB[nm1]; j = NB + idxB[nm2]
                val = -1j * kappa * sign  ## opposite sign on B ##
                H[i, j] += val
                H[j, i] += -np.conj(val)

    ## NN terms ##
    for (n,m) in idxA.keys():
        add_AB((n,m), (n+NN_SHIFTS["z"][0], m+NN_SHIFTS["z"][1]), Kz)
        add_AB((n,m), (n+NN_SHIFTS["x"][0], m+NN_SHIFTS["x"][1]), Kx)
        add_AB((n,m), (n+NN_SHIFTS["y"][0], m+NN_SHIFTS["y"][1]), Ky)

    ## NNN terms with Haldane orientation ##
    ## choose orientation signs based on the step index: +1 for the listed steps, and use both directions ##
    for (n,m) in idxA.keys():
        for dx,dy in NNN_STEPS:
            nm2 = (n+dx, m+dy)
            add_NNN("A", (n,m), nm2, +1)
            add_NNN("A", nm2, (n,m), +1)
    for (n,m) in idxB.keys():
        for dx,dy in NNN_STEPS:
            nm2 = (n+dx, m+dy)
            add_NNN("B", (n,m), nm2, +1)
            add_NNN("B", nm2, (n,m), +1)

    return H, idxA, idxB

def edge_mask(idxA, idxB, edge_thickness=0.7):
    ## heuristic edge mask: mark sites with fewer than 3 NN partners as edge ##
    ## build quick neighbor counts from NN_SHIFTS ##
    edgeA = {k: 0 for k in idxA.keys()}
    edgeB = {k: 0 for k in idxB.keys()}
    def hasB(nm, shift):
        return (nm[0]+shift[0], nm[1]+shift[1]) in idxB
    for nm in idxA.keys():
        cnt = 0
        for s in ["x","y","z"]:
            if hasB(nm, NN_SHIFTS[s]):
                cnt += 1
        edgeA[nm] = (cnt < 3)
    def hasA(nm, invshift):
        return (nm[0]-invshift[0], nm[1]-invshift[1]) in idxA
    for nm in idxB.keys():
        cnt = 0
        for s in ["x","y","z"]:
            if hasA(nm, NN_SHIFTS[s]):
                cnt += 1
        edgeB[nm] = (cnt < 3)
    return edgeA, edgeB

def metrics_from_state(psi, idxA, idxB, edgeA, edgeB):
    NA = len(idxA); NB = len(idxB)
    w = np.abs(psi)**2
    w = w / (np.sum(w) + 1e-16)
    wA = w[:NA]; wB = w[NA:]
    ipr = float(np.sum(w**2))

    left = 0.0; bulk = 0.0
    for (n,m), i in idxA.items():
        if edgeA[(n,m)]: left += float(wA[i])
        else: bulk += float(wA[i])
    for (n,m), j in idxB.items():
        if edgeB[(n,m)]: left += float(wB[j])
        else: bulk += float(wB[j])

    edge_participation = float(left)
    return ipr, edge_participation

def plot_ldos(psi, idxA, idxB, outpng, title):
    NA = len(idxA); NB = len(idxB)
    w = np.abs(psi)**2
    w = w / (np.sum(w) + 1e-16)
    xs, ys, cs = [], [], []

    for (n,m), i in idxA.items():
        pos = site_position("A", n, m)
        xs.append(pos[0]); ys.append(pos[1]); cs.append(w[i])
    for (n,m), j in idxB.items():
        pos = site_position("B", n, m)
        xs.append(pos[0]); ys.append(pos[1]); cs.append(w[NA+j])

    plt.figure(figsize=(6,5))
    sc = plt.scatter(xs, ys, c=cs, s=18, cmap="inferno")
    plt.colorbar(sc, label="|ψ|^2")
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()

def plot_spectrum(evals, outpng, title):
    plt.figure(figsize=(6.4,3.2))
    plt.plot(np.sort(np.real(evals)), ".", ms=4)
    plt.axhline(0, color="k", lw=1, ls=":")
    plt.ylabel("E")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Kitaev honeycomb quantum dot (finite island with open edges)")
    ap.add_argument("--Kx", type=float, default=1.0)
    ap.add_argument("--Ky", type=float, default=1.0)
    ap.add_argument("--Kz", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.06)
    ap.add_argument("--R", type=float, default=8.0)
    ap.add_argument("--Lx", type=int, default=16)
    ap.add_argument("--Ly", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="out_khoney_dot")
    ap.add_argument("--tag", type=str, default="dot")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    H, idxA, idxB = build_hamiltonian(args.Kx, args.Ky, args.Kz, args.kappa, args.R, args.Lx, args.Ly)
    evals, evecs = np.linalg.eigh(H)

    edgeA, edgeB = edge_mask(idxA, idxB)
    gap = float(np.partition(np.abs(evals), 1)[1])

    idx_min = int(np.argmin(np.abs(evals)))
    psi0 = evecs[:, idx_min]
    ipr, edge_part = metrics_from_state(psi0, idxA, idxB, edgeA, edgeB)

    plot_spectrum(evals, os.path.join(args.outdir, f"{args.tag}_spectrum.png"),
                  f"Spectrum (Kx={args.Kx}, Ky={args.Ky}, Kz={args.Kz}, κ={args.kappa}, R={args.R})")
    plot_ldos(psi0, idxA, idxB, os.path.join(args.outdir, f"{args.tag}_ldos.png"),
              f"In-gap state LDOS (closest to E=0)")

    res = {
        "N_sites": H.shape[0],
        "gap_proxy_minabsE": gap,
        "IPR_closest": ipr,
        "edge_participation_closest": edge_part
    }
    with open(os.path.join(args.outdir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps({
        "summary_json": os.path.join(args.outdir, f"{args.tag}_summary.json"),
        "spectrum_png": os.path.join(args.outdir, f"{args.tag}_spectrum.png"),
        "ldos_png": os.path.join(args.outdir, f"{args.tag}_ldos.png")
    }, indent=2))

if __name__ == "__main__":
    main()
