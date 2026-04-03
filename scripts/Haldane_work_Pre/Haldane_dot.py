#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

t = 1.0
t2 = 0.10
phi_haldane = np.pi/2
M = 0.20

R_list = [10, 12, 14, 16, 18, 20]
R_single = 15.0
edge_frac = 0.20
k_eigs = 120
epr_thresh = 0.5
seed = 0

DO_FLUX = False
flux_list = np.linspace(0.0, 0.5, 13)

np.random.seed(seed)

d1 = np.array([0.0, 1.0])
d2 = np.array([np.sqrt(3)/2, -0.5])
d3 = np.array([-np.sqrt(3)/2, -0.5])
NN_vecs = [d1, d2, d3]

a1 = d2 - d3
a2 = d1 - d3

b1 = d2 - d3
b2 = d3 - d1
b3 = d1 - d2
NNN_vecs = [b1, b2, b3]

def _key(pt, ndp=6):
    return (round(pt[0], ndp), round(pt[1], ndp))

def generate_dot_sites(R):
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
        k = _key(r); idx[k] = len(pos); pos.append(r); sub.append(+1)
    for r in B_pos:
        k = _key(r); idx[k] = len(pos); pos.append(r); sub.append(-1)
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

def build_haldane_hamiltonian(pos, sub, idx_map, t, t2, phi_haldane, M, phi_flux=0.0):
    N = len(pos)
    rows, cols, data = [], [], []

    for i in range(N):
        rows.append(i); cols.append(i); data.append(M if sub[i] == +1 else -M)

    for i in range(N):
        if sub[i] != +1:
            continue
        rA = pos[i]
        for d in NN_vecs:
            rB = rA + d
            j = idx_map.get(_key(rB))
            if j is not None and sub[j] == -1:
                phase = peierls_phase(rA, rB, phi_flux)
                val = t * phase
                rows += [i, j]
                cols += [j, i]
                data += [val, np.conj(val)]

    eiphi_A = np.exp(1j * phi_haldane)
    eiphi_B = np.exp(-1j * phi_haldane)
    for i in range(N):
        r = pos[i]
        base_phase = eiphi_A if sub[i] == +1 else eiphi_B
        for b in NNN_vecs:
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

def solve_near_zero(H, k, sigma=0.0):
    E, V = eigsh(H, k=k, sigma=sigma, which='LM')
    idx = np.argsort(E)
    return E[idx], V[:, idx]

def edge_participation(V, pos, R, edge_frac):
    r = np.linalg.norm(pos, axis=1)
    rim = r >= R * (1.0 - edge_frac)
    epr = []
    for j in range(V.shape[1]):
        psi2 = np.abs(V[:, j])**2
        epr.append(psi2[rim].sum() / psi2.sum())
    return np.array(epr)

def pick_edge_state(E, epr):
    i0 = np.argmin(np.abs(E))
    window = slice(max(i0-10, 0), min(i0+11, len(E)))
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
    return np.sum(np.abs(V)**4, axis=0)

def plot_spectrum(E, epr, title='Haldane dot spectrum (EPR colored)'):
    plt.figure(figsize=(6.2, 3.8))
    sc = plt.scatter(np.arange(len(E)), E, c=epr, s=12, cmap='viridis')
    plt.axhline(0.0, ls='--', lw=0.8, c='k')
    cb = plt.colorbar(sc); cb.set_label('edge participation ratio')
    plt.xlabel('eigenstate index (sorted)')
    plt.ylabel('energy')
    plt.title(title)
    plt.tight_layout()

def plot_ldos(pos, psi, R, Eval, title_prefix='LDOS'):
    psi2 = (np.abs(psi)**2)
    plt.figure(figsize=(5.2, 4.8))
    plt.scatter(pos[:,0], pos[:,1], c=psi2/psi2.max(), s=18, cmap='viridis')
    circ = plt.Circle((0,0), R, fill=False, ls='--', lw=0.8, color='k')
    ax = plt.gca(); ax.add_artist(circ); ax.set_aspect('equal', adjustable='box')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'{title_prefix} @ E={Eval:.4f}')
    plt.tight_layout()

def plot_spacing_vs_invR(R_list, Delta_list, v_edge_fit=None):
    invR = 1.0/np.array(R_list, dtype=float)
    plt.figure(figsize=(5.6, 3.8))
    plt.scatter(invR, Delta_list, s=28)
    if v_edge_fit is not None and np.isfinite(v_edge_fit):
        xs = np.linspace(invR.min()*0.9, invR.max()*1.05, 100)
        plt.plot(xs, v_edge_fit*xs, lw=1.5)
        plt.legend([r'$\Delta$ vs $1/R$', rf'fit: $\Delta \approx {v_edge_fit:.3f}/R$'])
    else:
        plt.legend([r'$\Delta$ vs $1/R$'])
    plt.xlabel(r'$1/R$')
    plt.ylabel(r'edge level spacing $\Delta$')
    plt.title('Edge spacing scales ~ $v_{\mathrm{edge}}/R$')
    plt.tight_layout()

def plot_spectral_flow(phi_list, E_flow):
    plt.figure(figsize=(6.2, 3.8))
    for row in E_flow:
        plt.plot(phi_list, row, lw=1.0)
    plt.xlabel(r'flux $\Phi/\Phi_0$')
    plt.ylabel('energy')
    plt.title('Spectral flow of low-lying levels (edge states move across gap)')
    plt.tight_layout()

def single_dot_demo(R=R_single, phi_flux=0.0):
    pos, sub, idx = generate_dot_sites(R)
    H = build_haldane_hamiltonian(pos, sub, idx, t, t2, phi_haldane, M, phi_flux=phi_flux)
    E, V = solve_near_zero(H, k=k_eigs, sigma=0.0)
    epr = edge_participation(V, pos, R, edge_frac)
    ipr = ipr_all(V)

    plot_spectrum(E, epr, title=f'Haldane dot R={R:.1f} (EPR colored)')
    j = pick_edge_state(E, epr)
    plot_ldos(pos, V[:, j], R, E[j], title_prefix='Edge-state LDOS')

    print(f'[single-dot] R={R:.1f}, sites={len(pos)}, min|E|={np.min(np.abs(E)):.4e}, mean EPR(in-gap?)={np.mean(epr[:10]):.3f}')
    return pos, sub, idx, E, V, epr, ipr

def size_sweep(R_list):
    Deltas = []
    for R in R_list:
        pos, sub, idx = generate_dot_sites(R)
        H = build_haldane_hamiltonian(pos, sub, idx, t, t2, phi_haldane, M, phi_flux=0.0)
        E, V = solve_near_zero(H, k=k_eigs, sigma=0.0)
        epr = edge_participation(V, pos, R, edge_frac)
        Delta = level_spacing_from_edge_ladder(E, epr, epr_thresh=epr_thresh)
        Deltas.append(Delta)
        print(f'[sweep] R={R:>4.1f}  sites={len(pos):>5d}  Δ≈{Delta:.5f}')
    invR = 1.0/np.array(R_list, dtype=float)
    Deltas = np.array(Deltas, dtype=float)
    mask = np.isfinite(Deltas)
    v_edge = np.nan
    if np.count_nonzero(mask) >= 2:
        v_edge = np.polyfit(invR[mask], Deltas[mask], 1)[0]
        print(f'[fit] edge velocity v_edge ≈ {v_edge:.5f} (from Δ ≈ v_edge / R)')
    plot_spacing_vs_invR(R_list, Deltas, v_edge_fit=v_edge)
    return Deltas, v_edge

def spectral_flow(R=R_single, k_keep=24, phi_list=flux_list):
    pos, sub, idx = generate_dot_sites(R)
    E_flow = []
    for phi_flux in phi_list:
        H = build_haldane_hamiltonian(pos, sub, idx, t, t2, phi_haldane, M, phi_flux=phi_flux)
        E, _ = solve_near_zero(H, k=max(k_keep, 2), sigma=0.0)
        E_flow.append(E[:k_keep])
        print(f'[flux] Φ/Φ0={phi_flux:.3f}  min|E|={np.min(np.abs(E)):.4e}')
    E_flow = np.array(E_flow).T
    plot_spectral_flow(phi_list, E_flow)
    return E_flow

if __name__ == '__main__':
    single_dot_demo(R=R_single, phi_flux=0.0)
    Deltas, v_edge = size_sweep(R_list)
    if DO_FLUX:
        spectral_flow(R=R_single, k_keep=24, phi_list=flux_list)
    plt.show()

    
import numpy as np
import matplotlib.pyplot as plt

# --- Haldane parameters (match these to your dot runs) ---
t = 1.0
t2 = 0.10
phi = np.pi/2
M = 0.20

# --- Honeycomb geometry (consistent with our earlier code) ---
d1 = np.array([0.0, 1.0])                   # A->B nearest neighbors
d2 = np.array([np.sqrt(3)/2, -0.5])
d3 = np.array([-np.sqrt(3)/2, -0.5])
NN = np.array([d1, d2, d3])

a1 = d2 - d3                                # Bravais vectors
a2 = d1 - d3

b1 = d2 - d3                                # next-nearest (same sublattice) differences
b2 = d3 - d1
b3 = d1 - d2
NNN = np.array([b1, b2, b3])

def energies_k(kx, ky, t=1.0, t2=0.1, M=0.0, phi=np.pi/2):
    f = np.sum(np.exp(1j*(kx*NN[:,0] + ky*NN[:,1])))  # off-diagonal
    gc = np.sum(np.cos(kx*NNN[:,0] + ky*NNN[:,1]))    # even NNN
    gs = np.sum(np.sin(kx*NNN[:,0] + ky*NNN[:,1]))    # odd  NNN
    d0 = 2.0*t2*np.cos(phi)*gc
    dz = M - 2.0*t2*np.sin(phi)*gs
    off2 = np.abs(t*f)**2
    E = d0 + np.array([+np.sqrt(off2 + dz*dz), -np.sqrt(off2 + dz*dz)])
    return np.sort(E)  # [E-, E+]

# --- reciprocal lattice from a1,a2 ---
A = np.column_stack([a1, a2])                # 2x2
B = 2*np.pi*np.linalg.inv(A).T               # columns are b1*, b2* (reciprocal)
g1, g2 = B[:,0], B[:,1]

# High-symmetry points (triangular lattice):
G = np.array([0.0, 0.0])
K = (g1 + 2*g2)/3.0
M = g1/2.0

# Build path Γ→K→M→Γ
def interpolate(p, q, n):
    return np.linspace(p, q, n, endpoint=False)

n_seg = 200
path = np.vstack([
    interpolate(G, K, n_seg),
    interpolate(K, M, n_seg),
    np.linspace(M, G, n_seg)                 # include endpoint on last segment
])

# Sample energies
Ek_lo, Ek_hi = [], []
for kx, ky in path:
    e = energies_k(kx, ky, t=t, t2=t2, M=M, phi=phi)
    Ek_lo.append(e[0]); Ek_hi.append(e[1])

Ek_lo = np.array(Ek_lo); Ek_hi = np.array(Ek_hi)

# x-axis with tick marks at Γ, K, M, Γ
x = np.arange(len(path))
ticks = [0, n_seg, 2*n_seg, 3*n_seg-1]
tick_labels = [r'$\Gamma$', r'$K$', r'$M$', r'$\Gamma$']

plt.figure(figsize=(6.4,3.8))
plt.plot(x, Ek_lo, lw=1.6)
plt.plot(x, Ek_hi, lw=1.6)
for tck in ticks:
    plt.axvline(tck, ls='--', lw=0.8, color='k', alpha=0.5)
plt.xticks(ticks, tick_labels)
plt.ylabel('Energy')
plt.title('Haldane model band structure (bulk)')
plt.tight_layout()
plt.show()
