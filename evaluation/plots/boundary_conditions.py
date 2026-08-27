#!/usr/bin/env python3
"""Generate the boundary-condition edge-density figures.

Two two-panel PDFs, one per model, each comparing a hard wall against a soft
wall on the same parent lattice:

    hubbard-psi2-Lx-vs-Ly-flake.pdf
    haldane-hubbard-psi2-Lx-vs-Ly-flake.pdf

The physics reproduces the standalone edge-diagnostic bundle that produced the
original single-panel PNGs (Haldane_Hubbard_Edge_Diagnostics_2026-08-20.zip):
a 20x20 parent lattice, an induced inner flake, and the summed site density of
the top-8 boundary-weighted eigenstates inside |E| <= 1. Reproduction is
checked against the values recorded in that bundle's manifest -- see CHECKS.

Invoked from generate.py; can also be run directly.
"""

from __future__ import annotations

import dataclasses

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.patches import Polygon

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- parameters
LATTICE = (20, 20)          # parent lattice, in unit cells
SITES_PER_CELL = 2          # A/B sublattice
SPIN = 2
T1 = 1.0                    # nearest-neighbour hopping (sets the energy unit)
T2 = 0.18                   # Haldane next-nearest-neighbour hopping
PHI = float(np.pi / 2.0)    # Haldane flux
M = 0.2                     # Semenoff mass
SOFT_V0 = 3.0               # soft-wall barrier height
EDGE_TOP_N = 8              # states summed, ranked by boundary participation
ENERGY_WINDOW = 1.0         # |E| window the ranked states are drawn from

# Plotting geometry. Only the drawn positions differ between the two models --
# the flake masks are cell-index based, so the physics is unaffected.
HONEYCOMB = dict(
    vectors=np.array([[-0.8660254037844386, -1.5], [0.8660254037844386, -1.5]]),
    sublattice=np.array([[0.0, 0.0], [0.0, -1.0]]),
)
SQUARE = dict(
    vectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
    sublattice=np.array([[0.0, 0.0], [0.354, 0.0]]),
)

N_SITES = LATTICE[0] * LATTICE[1] * SITES_PER_CELL


# ------------------------------------------------------------------ indexing
def site_index(cell, sub):
    return (cell[0] * LATTICE[1] + cell[1]) * SITES_PER_CELL + sub


def orbital(cell, sub, spin):
    return site_index(cell, sub) * SPIN + spin


def valid_cell(cell):
    return 0 <= cell[0] < LATTICE[0] and 0 <= cell[1] < LATTICE[1]


def positions(geom):
    xy = np.zeros((N_SITES, 2))
    for i in range(LATTICE[0]):
        for j in range(LATTICE[1]):
            origin = i * geom["vectors"][0] + j * geom["vectors"][1]
            for sub in range(SITES_PER_CELL):
                xy[site_index((i, j), sub)] = origin + geom["sublattice"][sub]
    return xy


def add_spin_hopping(H, cell_a, sub_a, cell_b, sub_b, amp):
    for spin in range(SPIN):
        a = orbital(cell_a, sub_a, spin)
        b = orbital(cell_b, sub_b, spin)
        H[a, b] += amp
        H[b, a] += np.conjugate(amp)


# ------------------------------------------------------------- Hamiltonians
def hubbard_hamiltonian():
    """Nearest-neighbour hopping only; U is not part of the single-particle sector."""
    H = np.zeros((N_SITES * SPIN, N_SITES * SPIN), dtype=complex)
    for i in range(LATTICE[0]):
        for j in range(LATTICE[1]):
            for di, dj in ((0, 0), (-1, 0), (0, -1)):
                target = (i + di, j + dj)
                if valid_cell(target):
                    add_spin_hopping(H, (i, j), 0, target, 1, -T1)
    return H


def haldane_hubbard_hamiltonian():
    """Haldane sector: Semenoff mass, NN hopping, complex NNN hopping."""
    H = np.zeros((N_SITES * SPIN, N_SITES * SPIN), dtype=complex)
    aa_amp = -T2 * np.exp(1j * PHI)
    bb_amp = -T2 * np.exp(-1j * PHI)
    for i in range(LATTICE[0]):
        for j in range(LATTICE[1]):
            cell = (i, j)
            for spin in range(SPIN):
                H[orbital(cell, 0, spin), orbital(cell, 0, spin)] += M
                H[orbital(cell, 1, spin), orbital(cell, 1, spin)] += -M
            for di, dj in ((0, 0), (-1, 0), (0, -1)):
                target = (i + di, j + dj)
                if valid_cell(target):
                    add_spin_hopping(H, cell, 0, target, 1, -T1)
            for di, dj in ((1, 0), (-1, 1), (0, -1)):
                target = (i + di, j + dj)
                if valid_cell(target):
                    add_spin_hopping(H, cell, 0, target, 0, aa_amp)
            for di, dj in ((-1, 0), (1, -1), (0, 1)):
                target = (i + di, j + dj)
                if valid_cell(target):
                    add_spin_hopping(H, cell, 1, target, 1, bb_amp)
    return H


# ----------------------------------------------------------- flake geometry
def flake_ranges(inner):
    sx = (LATTICE[0] - inner[0]) // 2
    sy = (LATTICE[1] - inner[1]) // 2
    return (sx, sx + inner[0]), (sy, sy + inner[1])


def flake_mask(inner):
    (sx, ex), (sy, ey) = flake_ranges(inner)
    mask = np.zeros(N_SITES, dtype=bool)
    for i in range(LATTICE[0]):
        for j in range(LATTICE[1]):
            active = sx <= i < ex and sy <= j < ey
            for sub in range(SITES_PER_CELL):
                mask[site_index((i, j), sub)] = active
    return mask


def flake_coordinate(inner):
    """Signed Chebyshev distance from the induced boundary, in cells."""
    (sx, ex), (sy, ey) = flake_ranges(inner)
    coord = np.zeros(N_SITES)
    for i in range(LATTICE[0]):
        for j in range(LATTICE[1]):
            value = max(sx - i - 0.5, i - (ex - 1) - 0.5, sy - j - 0.5, j - (ey - 1) - 0.5)
            for sub in range(SITES_PER_CELL):
                coord[site_index((i, j), sub)] = value
    return coord


# ------------------------------------------------------------------- solver
def site_density(eigvec):
    rho = np.real((np.abs(eigvec.reshape((-1, SPIN))) ** 2).sum(axis=1))
    total = float(rho.sum())
    return rho / total if total > 0 else rho


def density_to_full(density_active, active_mask):
    full = np.zeros(len(active_mask))
    full[active_mask] = density_active
    return full


def solve(H_base, inner, boundary, xi=None):
    coord = flake_coordinate(inner)
    edge_mask = np.abs(coord) <= 1.25
    hard_active = flake_mask(inner)

    if boundary == "hard_wall":
        active = hard_active
        omask = np.repeat(active, SPIN)
        H = H_base[np.ix_(omask, omask)]
    else:
        active = np.ones(N_SITES, dtype=bool)
        potential = 0.5 * SOFT_V0 * (1.0 + np.tanh(coord / float(xi)))
        H = H_base.copy()
        for site, value in enumerate(potential):
            for spin in range(SPIN):
                H[site * SPIN + spin, site * SPIN + spin] += value

    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.real(eigvals)

    edge_active = edge_mask[active]
    parts = np.array([site_density(eigvecs[:, k])[edge_active].sum()
                      for k in range(eigvecs.shape[1])])

    candidates = np.flatnonzero(np.abs(eigvals) <= ENERGY_WINDOW)
    if candidates.size < EDGE_TOP_N:
        candidates = np.argsort(np.abs(eigvals))[: EDGE_TOP_N * 3]
    ranked = sorted(candidates.tolist(),
                    key=lambda k: (parts[k], -abs(float(eigvals[k]))), reverse=True)
    indices = sorted(ranked[:EDGE_TOP_N])

    total = np.zeros(N_SITES)
    for k in indices:
        total += density_to_full(site_density(eigvecs[:, k]), active)
    total /= float(total.sum())

    return dict(density=total, active=active, hard_active=hard_active,
                indices=indices, e0=float(eigvals[np.argmin(np.abs(eigvals))]),
                n_active=int(active.sum()))


def convex_hull(points):
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    return half(pts)[:-1] + half(pts[::-1])[:-1]


# ------------------------------------------------------------------ styling
# Sizes are chosen so that, at the 490.176 pt width the other evaluation PDFs
# use, glyph heights match theirs: ticks ~14 pt, axis labels ~18.6 pt, bold
# panel labels ~17.5 pt (measured from evaluation/appendix-M-vs-U/E-M-vs-U.pdf).
FIG_W_IN = 490.176 / 72.0
TICK_FS = 7.4
LABEL_FS = 12.0
PANEL_FS = 8.7

_ttflist = font_manager.fontManager.ttflist
for _i, _entry in enumerate(_ttflist):
    if _entry.name.startswith("CMU ") and _entry.stretch != "normal":
        _ttflist[_i] = dataclasses.replace(_entry, stretch="normal")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "cmr10", "DejaVu Serif"],
    "font.sans-serif": ["CMU Sans Serif", "cmss10", "DejaVu Sans"],
    "font.monospace": ["CMU Typewriter Text", "cmtt10", "DejaVu Sans Mono"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})


def draw_panel(ax, cax, case, xy, label):
    active, density = case["active"], case["density"]

    inactive = ~active
    if np.any(inactive):
        ax.scatter(xy[inactive, 0], xy[inactive, 1], s=1.6,
                   color="#eeeeee", edgecolors="#bfbfbf", linewidths=0.12, zorder=2)

    vmax = float(density[active].max()) or 1.0
    sizes = 1.4 + 40.0 * density[active] / vmax
    sc = ax.scatter(xy[active, 0], xy[active, 1], c=density[active], s=sizes,
                    cmap="magma", edgecolors="#202020", linewidths=0.10, zorder=3)

    ax.add_patch(Polygon(convex_hull(xy[case["hard_active"]]), fill=False,
                         color="#1f77b4", linestyle="--", linewidth=1.0,
                         alpha=0.95, zorder=5))

    cbar = ax.figure.colorbar(sc, cax=cax)
    cbar.set_label(r"$|\psi_i|^2$", fontsize=LABEL_FS, labelpad=2)
    cbar.ax.tick_params(labelsize=TICK_FS, length=2, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.5)

    ax.set_xlabel(r"$L_x$", fontsize=LABEL_FS, labelpad=1.5)
    ax.set_ylabel(r"$L_y$", fontsize=LABEL_FS, labelpad=1.5)
    ax.tick_params(labelsize=TICK_FS, length=2, width=0.5, pad=1.5)
    ax.set_aspect("equal", adjustable="box")
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    pad = max(0.8, 0.04 * span)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.34)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title(label, fontsize=PANEL_FS, fontweight="bold", loc="left", pad=4)


def build_figure(cases, labels, xy, out_name, height_in):
    fig = plt.figure(figsize=(FIG_W_IN, height_in))
    # [panel a][cbar a][spacer][panel b][cbar b] -- the spacer keeps the first
    # colorbar's label clear of the second panel's y axis.
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.035, 0.46, 1.0, 0.035],
                          wspace=0.09, left=0.05, right=0.96, top=0.92, bottom=0.10)
    pairs = []
    for k, (case, label) in enumerate(zip(cases, labels)):
        col = 0 if k == 0 else 3
        ax = fig.add_subplot(gs[0, col])
        cax = fig.add_subplot(gs[0, col + 1])
        draw_panel(ax, cax, case, xy, label)
        pairs.append((ax, cax))
    # the panels carry an equal aspect, so their boxes shrink at draw time;
    # pin each colorbar to the height its panel actually ends up with.
    fig.canvas.draw()
    for ax, cax in pairs:
        panel = ax.get_position()
        bar = cax.get_position()
        cax.set_position([bar.x0, panel.y0, bar.width, panel.height])
    out = OUT / out_name
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- CHECKS
# Recorded in the diagnostic bundle's manifest.json / the original PNG
# annotations. Reproduction is asserted so the figures cannot silently drift.
CHECKS = {
    ("hubbard", "hard"): (-8.045576e-07, [284, 285, 286, 287, 288, 289, 290, 291], 288),
    ("hubbard", "soft"): (-0.01116258, [272, 273, 274, 275, 284, 285, 290, 291], 800),
    ("haldane_hubbard", "hard"): (0.032273193, [112, 113, 126, 127, 128, 129, 142, 143], 128),
    ("haldane_hubbard", "soft"): (0.012734610, [100, 101, 108, 109, 112, 113, 130, 131], 800),
}


def check(model, wall, case):
    e0, indices, n_active = CHECKS[(model, wall)]
    assert case["indices"] == indices, f"{model}/{wall}: edge states {case['indices']} != {indices}"
    assert case["n_active"] == n_active, f"{model}/{wall}: active {case['n_active']} != {n_active}"
    # the near-zero state is one of a +/- degenerate pair; compare magnitudes
    assert abs(abs(case["e0"]) - abs(e0)) < 1e-6, f"{model}/{wall}: E0 {case['e0']} != {e0}"
    print(f"  ok  {model:16s} {wall:4s} wall  E0={case['e0']:+.6g}  active={case['n_active']}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Hubbard (inner 12x12, soft-wall xi=0.5)")
    H = hubbard_hamiltonian()
    hub = [solve(H, (12, 12), "hard_wall"), solve(H, (12, 12), "soft_wall", xi=0.5)]
    check("hubbard", "hard", hub[0])
    check("hubbard", "soft", hub[1])
    p1 = build_figure(hub, ["a. Hard wall", "b. Soft wall"],
                      positions(SQUARE), "hubbard-psi2-Lx-vs-Ly-flake.pdf", 2.7)

    print("Haldane-Hubbard (inner 8x8, soft-wall xi=0.1)")
    H = haldane_hubbard_hamiltonian()
    hh = [solve(H, (8, 8), "hard_wall"), solve(H, (8, 8), "soft_wall", xi=0.1)]
    check("haldane_hubbard", "hard", hh[0])
    check("haldane_hubbard", "soft", hh[1])
    p2 = build_figure(hh, ["a. Hard wall", "b. Soft wall"],
                      positions(HONEYCOMB), "haldane-hubbard-psi2-Lx-vs-Ly-flake.pdf", 5.5)

    for p in (p1, p2):
        print("wrote", p)


if __name__ == "__main__":
    main()
