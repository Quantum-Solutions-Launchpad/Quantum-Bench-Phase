#!/usr/bin/env python3
"""Regenerate every manuscript figure under ``figures/evaluation`` of paper-qbp.

Reads the recorded runs in ``evaluation/`` and writes one PDF per manuscript
figure straight into ``evaluation/plots/``, named
``<model>-<observable>-<x>-vs-<y>[-<extra>]``:

    haldane-E-M-vs-phi.pdf                  (was M-vs-phi/E-M-vs-phi)
    haldane-E-M-vs-U.pdf                    (was appendix-M-vs-U/E-M-vs-U)
    haldane-E-N_occ-vs-t2.pdf               (was nocc-vs-t2/E-nocc-vs-t2)
    haldane-E-kx-vs-ky.pdf                  (was band-structure/E-kx-vs-ky)
    haldane-hubbard-psi2-Lx-vs-Ly-flake.pdf (was psi2-haldane-hubbard-flake)
    hubbard-E-vs-N_occ-hardware.pdf         (was hardware/hubbard/E-n_occ-hardware)
    hubbard-S-N_occ-vs-U.pdf                (was Magnetization/M-n_occ-vs-U)
    hubbard-psi2-Lx-vs-Ly-flake.pdf         (was psi2-hubbard-flake)
    max3sat-n_viol-vs-alpha-hardware.pdf    (was hardware/max3sat/n_viol-ratio-hardware)
    tfim-E-Lx-vs-h.pdf                      (was tfim/E-Lx-vs-h)

Usage:  python3 evaluation/plots/generate.py [figure ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL = Path(__file__).resolve().parent.parent
OUT = EVAL / "plots"

# The manuscript PDFs were rendered with this matplotlib; other versions shift
# text metrics (legend row heights in particular) by a fraction of a point.
MPL_VERSION = "3.10.8"

FIG_W = 490.176 / 72.0

CM_SERIF = ["CMU Serif", "cmr10", "DejaVu Serif"]
CM_SANS = ["CMU Sans Serif", "cmss10", "DejaVu Sans"]
CM_MONO = ["CMU Typewriter Text", "cmtt10", "DejaVu Sans Mono"]

SPINE = "0.6"
SPINE_LW = 0.9
TICK_COLOR = "#888888"
TICK_LEN = 2.4
TICK_LW = 0.8
TICK_PAD = 1.8
LABEL_PAD = 1.5
GRID_KW = dict(linestyle="--", linewidth=0.5, alpha=0.35)
LOCATOR_STEPS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 10]
CBAR_STEPS = [1, 2, 2.5, 5, 10]
MAIN_NBINS = 5
ERR_NBINS = 3
HEATMAP_NBINS = 4

MAIN_LABEL_SIZE = 16
MAIN_TICK_SIZE = 12
ERR_LABEL_SIZE = 13.5
ERR_TICK_SIZE = 10.5
LEGEND_SIZE = 11

MARKER_SIZE = 4.8674942218
MARKER_EDGE = "#4a4a4a"
MARKER_EDGE_LW = 0.4

EXACT_COLOR = "#3b0f70"
EXACT_LW = 1.5
NOMIT_COLOR = "#9ecae1"
MIT_COLOR = "#08519c"

BAR_NOMIT = (0.9882352941, 0.6564090734, 0.5431910804)
BAR_MIT = (0.5412226067, 0.0332179931, 0.0686966551)
BAR_EDGE = "#3a3a3a"
BAR_EDGE_LW = 0.5
BAR_WIDTH = 0.39

VQE_COLOR = "#0072B2"
IQPE_COLOR = "#6DBF82"
DMRG_COLOR = "#D7277C"
VQE_SIZE = 13
IQPE_SIZE = 15
DMRG_SIZE = 12
PANE_COLOR = "#cccccc"
AX3D_LABELPAD = (3, 3, 4)
AX3D_ELEV = 24
AX3D_AZIM = -58
AX3D_BOX_ASPECT = (1, 1, 0.84)
AX3D_ZOOM = 1.1
SURFACE_ALPHA = 0.12
SURFACE_LW = 0.9
SURFACE_LINE_ALPHA = 0.95
SURFACE_DOT = 20
AX3D_TICK_PAD = -1.0
AX3D_NBINS = 4
AX3D_STEPS = [1, 1.5, 2, 3, 4, 5, 6, 10]

TOP_LEGEND_SIZE = 12.5
TOP_LEGEND_KW = dict(
    loc="upper center",
    borderaxespad=0.2,
    borderpad=0.3,
    handlelength=1.9,
    handletextpad=0.35,
    columnspacing=0.8,
    framealpha=0.8,
    edgecolor=PANE_COLOR,
    facecolor="white",
)

CBAR_WIDTH = 10.08
CBAR_GAP = 5.04
CBAR_TICK_LEN = 1.8
CBAR_TICK_LW = 0.6
CBAR_TICK_PAD = 1.2
CBAR_OUTLINE_LW = 0.7
CBAR_LABEL_PAD = 2.5
CBAR_MAIN_NBINS = 4
CBAR_ERR_NBINS = 3

LEGEND_KW = dict(
    fontsize=LEGEND_SIZE,
    borderaxespad=0.4,
    borderpad=0.3,
    handlelength=1.6,
    handletextpad=0.4,
    labelspacing=0.3,
    framealpha=0.8,
    edgecolor=SPINE,
    facecolor="white",
)


def magma_dark():
    return LinearSegmentedColormap.from_list(
        "magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256)))


def reds_dark():
    return LinearSegmentedColormap.from_list(
        "reds_dark", plt.cm.Reds(np.linspace(0.07, 1.0, 256)))


def edges(a):
    a = np.asarray(a, dtype=float)
    if len(a) == 1:
        return np.array([a[0] - 0.5, a[0] + 0.5])
    d = np.diff(a) / 2.0
    return np.concatenate([[a[0] - d[0]], a[:-1] + d, [a[-1] + d[-1]]])


def momentum_ticks(ax, axis, vals):
    arr = np.asarray(vals, dtype=float)
    lo = int(np.floor(arr.min() / np.pi + 1e-6))
    hi = int(np.ceil(arr.max() / np.pi - 1e-6))
    ticks = np.arange(lo, hi + 1) * np.pi

    def fmt(t):
        if abs(t) < 1e-9:
            return r"$0$"
        if abs(t - np.pi) < 1e-9:
            return r"$\pi$"
        if abs(t + np.pi) < 1e-9:
            return r"$-\pi$"
        return rf"${int(round(t / np.pi))}\pi$"

    getattr(ax, "set_%sticks" % axis)(ticks)
    getattr(ax, "set_%sticklabels" % axis)([fmt(t) for t in ticks])


def unstretch_cm_faces():
    import dataclasses

    from matplotlib import font_manager

    fonts = font_manager.fontManager.ttflist
    for i, entry in enumerate(fonts):
        if entry.name.startswith("CMU ") and entry.stretch != "normal":
            fonts[i] = dataclasses.replace(entry, stretch="normal")


def rcparams():
    unstretch_cm_faces()
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": CM_SERIF,
        "font.sans-serif": CM_SANS,
        "font.monospace": CM_MONO,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": True,
        "figure.dpi": 300,
    })


def style_axes(ax, label_size, tick_size, nbins):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(SPINE_LW)
    ax.tick_params(direction="out", length=TICK_LEN, width=TICK_LW,
                   color=TICK_COLOR, labelsize=tick_size, pad=TICK_PAD)
    ax.grid(True, **GRID_KW)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=LOCATOR_STEPS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=LOCATOR_STEPS))
    ax.set_axisbelow(True)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.xaxis.labelpad = LABEL_PAD
    ax.yaxis.labelpad = LABEL_PAD


def style_heatmap_axes(ax, label_size, tick_size, nbins=None):
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(SPINE_LW)
    ax.tick_params(direction="out", length=TICK_LEN, width=TICK_LW,
                   color=TICK_COLOR, labelsize=tick_size, pad=TICK_PAD)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.xaxis.labelpad = LABEL_PAD
    ax.yaxis.labelpad = LABEL_PAD
    if nbins is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=LOCATOR_STEPS))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=LOCATOR_STEPS))


def style_colorbar(cbar, label, label_size, tick_size, nbins):
    cbar.locator = MaxNLocator(nbins=nbins, steps=CBAR_STEPS)
    cbar.update_ticks()
    cbar.outline.set_edgecolor(SPINE)
    cbar.outline.set_linewidth(CBAR_OUTLINE_LW)
    cbar.ax.tick_params(length=CBAR_TICK_LEN, width=CBAR_TICK_LW,
                        pad=CBAR_TICK_PAD, labelsize=tick_size)
    cbar.set_label(label, size=label_size, labelpad=CBAR_LABEL_PAD)


class GradientHandler(HandlerBase):
    """Legend handle for the analytic surface: a colormap bar with a dot."""

    def __init__(self, cmap):
        self.cmap = cmap
        super().__init__()

    def create_artists(self, _legend, _handle, xdescent, ydescent, width, height,
                       _fontsize, trans):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        line_h = height * 0.18
        line_y = ydescent + (height - line_h) / 2
        image = BboxImage(TransformedBbox(Bbox.from_bounds(xdescent, line_y,
                                                           width, line_h), trans),
                          cmap=self.cmap)
        image.set_data(gradient)
        image.set_alpha(0.9)
        dot = Line2D([xdescent + width * 0.5], [ydescent + height * 0.5],
                     marker="o", markersize=7.5, linestyle="none",
                     markerfacecolor=self.cmap(0.5), markeredgewidth=0)
        dot.set_transform(trans)
        return [image, dot]


def marker_handle(color, marker, label, size=8.5):
    return Line2D([0], [0], color="w", marker=marker, markersize=size,
                  markerfacecolor=color, label=label)


def top_legend(fig, handles, handler_map=None):
    leg = fig.legend(handles=handles, ncol=len(handles),
                     fontsize=TOP_LEGEND_SIZE, handler_map=handler_map,
                     **TOP_LEGEND_KW)
    leg.get_frame().set_linewidth(0.7)
    return leg


def legend(ax, handles, loc):
    leg = ax.legend(handles=handles, loc=loc, **LEGEND_KW)
    leg.get_frame().set_linewidth(0.7)
    return leg


def save(fig, relpath):
    path = OUT / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def load(*parts):
    with open(EVAL.joinpath(*parts)) as fh:
        return json.load(fh)


def series(data, method):
    res = data["result"][method]
    return np.array([res[str(i)] for i in range(len(res))], dtype=float)


# --------------------------------------------------------------------------
# hardware comparisons
# --------------------------------------------------------------------------

def _hardware_figure(xs, exact, nomit, mit, xlabel, ylabel, relpath):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(FIG_W, 229.248 / 72.0))
    axL = fig.add_axes([41.76 / 490.176, 36 / 229.248,
                        187.488 / 490.176, 187.488 / 229.248])
    axR = fig.add_axes([292.608 / 490.176, 36 / 229.248,
                        187.488 / 490.176, 187.488 / 229.248])

    style_axes(axL, MAIN_LABEL_SIZE, MAIN_TICK_SIZE, MAIN_NBINS)
    style_axes(axR, ERR_LABEL_SIZE, ERR_TICK_SIZE, ERR_NBINS)

    axL.plot(xs, exact, color=EXACT_COLOR, lw=EXACT_LW, zorder=3)
    for vals, color in ((nomit, NOMIT_COLOR), (mit, MIT_COLOR)):
        axL.plot(xs, vals, ls="none", marker="o", markersize=MARKER_SIZE,
                 markerfacecolor=color, markeredgecolor=MARKER_EDGE,
                 markeredgewidth=MARKER_EDGE_LW, zorder=4)
    axL.set_xlabel(xlabel)
    axL.set_ylabel(ylabel)
    lo = min(exact.min(), nomit.min(), mit.min())
    hi = max(exact.max(), nomit.max(), mit.max())
    pad = 0.08 * (hi - lo)
    axL.set_ylim(lo - pad, hi + pad)
    xpad = 0.06 * (xs[-1] - xs[0])
    axL.set_xlim(xs[0] - xpad, xs[-1] + xpad)

    handles = [
        Line2D([0], [0], color=EXACT_COLOR, lw=1.8, label="Exact"),
        Line2D([0], [0], color="w", marker="o", markersize=8,
               markerfacecolor=NOMIT_COLOR, markeredgecolor=MARKER_EDGE,
               markeredgewidth=MARKER_EDGE_LW, label="No mitigation"),
        Line2D([0], [0], color="w", marker="o", markersize=8,
               markerfacecolor=MIT_COLOR, markeredgecolor=MARKER_EDGE,
               markeredgewidth=MARKER_EDGE_LW, label="With mitigation"),
    ]
    legend(axL, handles, "upper left")

    err_nomit = np.abs(nomit - exact)
    err_mit = np.abs(mit - exact)
    axR.bar(xs - BAR_WIDTH / 2, err_nomit, width=BAR_WIDTH, color=BAR_NOMIT,
            edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, label="No mitigation")
    axR.bar(xs + BAR_WIDTH / 2, err_mit, width=BAR_WIDTH, color=BAR_MIT,
            edgecolor=BAR_EDGE, linewidth=BAR_EDGE_LW, label="With mitigation")
    axR.set_xlabel(xlabel)
    axR.set_ylabel("abs. err.")
    axR.set_xlim(xs[0] - 0.53, xs[-1] + 0.53)
    axR.set_ylim(0.0, 1.42 * max(err_nomit.max(), err_mit.max()))
    axR.set_xticks(xs)
    legend(axR, [Patch(facecolor=BAR_NOMIT, edgecolor=BAR_EDGE,
                       linewidth=BAR_EDGE_LW, label="No mitigation"),
                 Patch(facecolor=BAR_MIT, edgecolor=BAR_EDGE,
                       linewidth=BAR_EDGE_LW, label="With mitigation")],
           "upper right")

    save(fig, relpath)


def fig_hardware_hubbard():
    exact_data = load("hardware", "hubbard", "analytic-n_occ-U3.json")
    nomit_data = load("hardware", "hubbard", "iqm-garnet-nomit-vqe-n_occ-U3.json")
    mit_data = load("hardware", "hubbard", "iqm-garnet-vqe-n_occ-U3.json")
    xs = np.array(exact_data["x_values"], dtype=float)
    _hardware_figure(xs, series(exact_data, "analytic"),
                     series(nomit_data, "vqe"), series(mit_data, "vqe"),
                     r"$N_\mathrm{occ}$", "$E$",
                     "hubbard-E-vs-N_occ-hardware.pdf")


def _n_clauses(data):
    n = int(re.search(r"_n-(\d+)_", data["parameters"]["keys"][0]).group(1))
    return n * np.array(data["x_values"], dtype=float)


def fig_hardware_max3sat():
    nomit_data = load("hardware", "max3sat", "iqm-garnet-nomit-vqe-ratio-n4.json")
    mit_data = load("hardware", "max3sat", "iqm-garnet-vqe-ratio-n4.json")
    xs = np.array(nomit_data["x_values"], dtype=float)
    clauses = _n_clauses(nomit_data)
    exact = clauses - np.round(series(nomit_data, "analytic"))
    _hardware_figure(xs, exact,
                     clauses - series(nomit_data, "vqe"),
                     clauses - series(mit_data, "vqe"),
                     r"$\alpha$", r"$n_\mathrm{viol}$",
                     "max3sat-n_viol-vs-alpha-hardware.pdf")


# --------------------------------------------------------------------------
# M vs phi
# --------------------------------------------------------------------------

def grid(data, method):
    nx, ny = len(data["x_values"]), len(data["y_values"])
    out = np.full((nx, ny), np.nan)
    for xi, row in data["result"][method].items():
        for yi, value in row.items():
            out[int(xi), int(yi)] = value
    return out


def heatmap(ax, cax, x_vals, y_vals, Z, cmap, vmin, vmax):
    mesh = ax.pcolormesh(edges(x_vals), edges(y_vals), Z.T, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    ax.set_xlim(edges(x_vals)[0], edges(x_vals)[-1])
    ax.set_ylim(edges(y_vals)[0], edges(y_vals)[-1])
    return ax.get_figure().colorbar(mesh, cax=cax)


def fig_m_vs_phi():
    analytic = load("M-vs-phi", "simulated-ideal-analytic-E-M-vs-phi.json")
    vqe = load("M-vs-phi", "simulated-ideal-vqe-E-M-vs-phi.json")
    dmrg = load("M-vs-phi", "simulated-ideal-dmrg-E-M-vs-phi.json")
    phi = np.array(analytic["x_values"], dtype=float)
    M = np.array(analytic["y_values"], dtype=float)
    A = grid(analytic, "analytic")
    V = grid(vqe, "vqe")
    D = grid(dmrg, "dmrg")

    W, H = 490.176, 219.744
    fig = plt.figure(figsize=(W / 72, H / 72))

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    ax = fig.add_axes(rect(41.76, 36, 177.984, 177.984))
    cax = fig.add_axes(rect(224.784, 36, CBAR_WIDTH, 177.984))
    style_heatmap_axes(ax, MAIN_LABEL_SIZE, MAIN_TICK_SIZE, HEATMAP_NBINS)
    cbar = heatmap(ax, cax, phi, M, V, magma_dark(), np.nanmin(V), np.nanmax(V))
    momentum_ticks(ax, "x", phi)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("$M$")
    style_colorbar(cbar, "VQE $E$", MAIN_TICK_SIZE, ERR_TICK_SIZE, CBAR_MAIN_NBINS)

    for bottom, err, name in ((129.312, np.abs(V - A), "VQE"),
                              (36.0, np.abs(D - A), "DMRG")):
        axe = fig.add_axes(rect(329.184, bottom, 84.672, 84.672))
        caxe = fig.add_axes(rect(418.896, bottom, CBAR_WIDTH, 84.672))
        style_heatmap_axes(axe, ERR_LABEL_SIZE, ERR_TICK_SIZE, ERR_NBINS)
        cb = heatmap(axe, caxe, phi, M, err, reds_dark(), 0.0, np.nanmax(err))
        momentum_ticks(axe, "x", phi)
        if bottom == 36.0:
            axe.set_xlabel(r"$\phi$")
        else:
            axe.set_xticklabels([])
        axe.set_ylabel("$M$")
        style_colorbar(cb, name + "\nabs. err.", LEGEND_SIZE, ERR_TICK_SIZE,
                       CBAR_ERR_NBINS)

    save(fig, "haldane-E-M-vs-phi.pdf")


# --------------------------------------------------------------------------
# magnetization
# --------------------------------------------------------------------------

def resample_y(A, y_src, y_dst):
    out = np.empty((A.shape[0], len(y_dst)))
    for i in range(A.shape[0]):
        out[i] = np.interp(y_dst, y_src, A[i])
    return out


def _magnetization_block(fig, rect, base, analytic_data, vqe_data, symbol,
                         u_max=None):
    n_occ = np.array(analytic_data["x_values"], dtype=float)
    U_a = np.array(analytic_data["y_values"], dtype=float)
    U_v = np.array(vqe_data["y_values"], dtype=float)
    A = grid(analytic_data, "analytic")
    V = grid(vqe_data, "vqe")

    err = np.abs(V - resample_y(A, U_a, U_v))

    keep = U_a <= (U_v.max() if u_max is None else u_max)
    U_a, A = U_a[keep], A[:, keep]

    ax = fig.add_axes(rect(41.76, base, 177.984, 177.984))
    cax = fig.add_axes(rect(224.784, base, CBAR_WIDTH, 177.984))
    style_heatmap_axes(ax, MAIN_LABEL_SIZE, MAIN_TICK_SIZE, HEATMAP_NBINS)
    cbar = heatmap(ax, cax, n_occ, U_a, A, magma_dark(),
                   np.nanmin(A), np.nanmax(A))
    ax.set_xlabel(r"$N_\mathrm{occ}$")
    ax.set_ylabel("$U$")
    style_colorbar(cbar, "Analytic " + symbol, MAIN_TICK_SIZE, ERR_TICK_SIZE,
                   CBAR_MAIN_NBINS)

    panels = ((base + 93.312, V, magma_dark(), np.nanmin(V), np.nanmax(V), symbol),
              (base, err, reds_dark(), 0.0, np.nanmax(err), "abs. err."))
    for bottom, Z, cmap, vmin, vmax, name in panels:
        axe = fig.add_axes(rect(329.184, bottom, 84.672, 84.672))
        caxe = fig.add_axes(rect(418.896, bottom, CBAR_WIDTH, 84.672))
        style_heatmap_axes(axe, ERR_LABEL_SIZE, ERR_TICK_SIZE, ERR_NBINS)
        cb = heatmap(axe, caxe, n_occ, U_v, Z, cmap, vmin, vmax)
        if bottom == base:
            axe.set_xlabel(r"$N_\mathrm{occ}$")
        else:
            axe.set_xticklabels([])
        axe.set_ylabel("$U$")
        style_colorbar(cb, "VQE\n" + name, LEGEND_SIZE, ERR_TICK_SIZE,
                       CBAR_ERR_NBINS)


def fig_magnetization():
    W, H = 490.176, 475.488
    fig = plt.figure(figsize=(W / 72, H / 72))

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    blocks = ((291.744, "S_stag", "M_stag", r"$S_\mathrm{stag}$", None),
              (36.0, "S_total", "M_total", r"$S_\mathrm{tot}$", 60.0))
    for base, analytic_key, vqe_key, symbol, u_max in blocks:
        analytic_data = load(
            "magnetization", analytic_key,
            "simulated-ideal-analytic-%s-n_occ-vs-U.json" % analytic_key)
        vqe_data = load("magnetization", vqe_key,
                        "vqe-%s-heatmap-n_occ-vs-U.json" % vqe_key)
        _magnetization_block(fig, rect, base, analytic_data, vqe_data, symbol,
                             u_max)

    save(fig, "hubbard-S-N_occ-vs-U.pdf")


# --------------------------------------------------------------------------
# band structure
# --------------------------------------------------------------------------

def band_grid(data, method):
    nx, ny = len(data["x_values"]), len(data["y_values"])
    first = next(iter(next(iter(data["result"][method].values())).values()))
    nb = len(first) if isinstance(first, list) else 1
    out = np.full((nx, ny, nb), np.nan)
    for xi, row in data["result"][method].items():
        for yi, value in row.items():
            out[int(xi), int(yi)] = value
    return out if isinstance(first, list) else out[:, :, 0]


def surface3d(ax, x_vals, y_vals, Z):
    """The analytic reference: a translucent magma surface with one line and
    one dot row per y value (repeated per band for 3-D band structures)."""
    cmap = magma_dark()
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    ny = len(y_vals)
    layers = [Z] if Z.ndim == 2 else [Z[:, :, b] for b in range(Z.shape[-1])]
    for layer in layers:
        ax.plot_surface(X, Y, layer, cmap=cmap, alpha=SURFACE_ALPHA,
                        edgecolor="none", rcount=ny, ccount=len(x_vals))
        for iy, yv in enumerate(y_vals):
            color = cmap(iy / max(ny - 1, 1))
            ax.plot(x_vals, [yv] * len(x_vals), layer[:, iy], color=color,
                    linewidth=SURFACE_LW, alpha=SURFACE_LINE_ALPHA, zorder=4)
            ax.scatter(x_vals, [yv] * len(x_vals), layer[:, iy], color=color,
                       s=SURFACE_DOT, alpha=0.4, depthshade=False, zorder=5)
    return X, Y


def scatter3d(ax, X, Y, series, z_clip):
    for values, color, marker, size in series:
        flat = np.asarray(values, dtype=float).ravel()
        mask = flat >= z_clip
        ax.scatter(X.ravel()[mask], Y.ravel()[mask], flat[mask], color=color,
                   marker=marker, s=size, depthshade=True, zorder=6)
    ax.set_zlim(bottom=z_clip)


def style_axes3d(ax, label_size, tick_size):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(PANE_COLOR)
    ax.view_init(elev=AX3D_ELEV, azim=AX3D_AZIM)
    ax.set_box_aspect(AX3D_BOX_ASPECT, zoom=AX3D_ZOOM)
    ax.tick_params(labelsize=tick_size, pad=AX3D_TICK_PAD)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_size(label_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=AX3D_NBINS, steps=AX3D_STEPS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=AX3D_NBINS, steps=AX3D_STEPS))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=AX3D_NBINS))


def fig_band_structure():
    analytic = load("band-structure", "simulated-noisy-analytic-E-kx-vs-ky.json")
    vqe = load("band-structure", "simulated-noisy-vqe-E-kx-vs-ky.json")
    iqpe = load("band-structure",
                "simulated-noisy-iqpe-E-kx-vs-ky-filtered.json")
    dmrg = load("band-structure", "simulated-noisy-dmrg-E-kx-vs-ky.json")
    kx = np.array(analytic["x_values"], dtype=float)
    ky = np.array(analytic["y_values"], dtype=float)
    bands = band_grid(analytic, "analytic")
    A0 = bands[:, :, 0]
    V = band_grid(vqe, "vqe")
    I = band_grid(iqpe, "iqpe")
    D = band_grid(dmrg, "dmrg")

    W, H = 490.176, 279.264
    fig = plt.figure(figsize=(W / 72, H / 72))

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    ax = fig.add_axes(rect(10.08, 36, 214.464, 214.464), projection="3d")
    style_axes3d(ax, MAIN_LABEL_SIZE, ERR_TICK_SIZE)
    X, Y = surface3d(ax, kx, ky, bands)
    scatter3d(ax, X, Y, ((D, DMRG_COLOR, "s", DMRG_SIZE),
                         (V, VQE_COLOR, "o", VQE_SIZE),
                         (I, IQPE_COLOR, "^", IQPE_SIZE)),
              float(np.nanmin(bands)))
    momentum_ticks(ax, "x", kx)
    momentum_ticks(ax, "y", ky)
    ax.set_xlabel("$k_x$", labelpad=AX3D_LABELPAD[0])
    ax.set_ylabel("$k_y$", labelpad=AX3D_LABELPAD[1])
    ax.set_zlabel(r"$E(\mathbf{k})$", labelpad=AX3D_LABELPAD[2])

    for bottom, err, name in ((147.552, np.abs(V - A0), "VQE"),
                              (36.0, np.abs(I - A0), "IQPE")):
        axe = fig.add_axes(rect(310.944, bottom, 102.912, 102.912))
        caxe = fig.add_axes(rect(418.896, bottom, CBAR_WIDTH, 102.912))
        style_heatmap_axes(axe, ERR_LABEL_SIZE, ERR_TICK_SIZE, ERR_NBINS)
        cb = heatmap(axe, caxe, kx, ky, err, reds_dark(), 0.0, np.nanmax(err))
        momentum_ticks(axe, "x", kx)
        momentum_ticks(axe, "y", ky)
        if bottom == 36.0:
            axe.set_xlabel("$k_x$")
        else:
            axe.set_xticklabels([])
        axe.set_ylabel("$k_y$")
        style_colorbar(cb, name + "\nabs. err.", LEGEND_SIZE, ERR_TICK_SIZE,
                       CBAR_ERR_NBINS)

    surface = mpatches.Patch(label="Analytic bands")
    top_legend(fig, [surface,
                     marker_handle(VQE_COLOR, "o", "VQE"),
                     marker_handle(IQPE_COLOR, "^", "IQPE"),
                     marker_handle(DMRG_COLOR, "s", "DMRG")],
               handler_map={surface: GradientHandler(magma_dark())})

    save(fig, "haldane-E-kx-vs-ky.pdf")


# --------------------------------------------------------------------------
# stacked 3-D + error-heatmap figures (tfim, nocc-vs-t2, appendix M-vs-U)
# --------------------------------------------------------------------------

CAPTION_SIZE = 15


def caption(fig, W, H, x, y, text):
    fig.text(x / W, y / H, text, fontsize=CAPTION_SIZE, fontweight="bold",
             ha="left", va="baseline")


def stacked_figure(W, H, blocks, ax3d_rect, panel_rect, panel_pitch,
                   caption_dy, legend_handles, handler_map, relpath,
                   labels3d, panel_labels, labelpad3d=None):
    fig = plt.figure(figsize=(W / 72, H / 72))

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    for base, title, x_vals, y_vals, Z, series, panels in blocks:
        ax = fig.add_axes(rect(ax3d_rect[0], base + ax3d_rect[1],
                               ax3d_rect[2], ax3d_rect[3]), projection="3d")
        style_axes3d(ax, MAIN_LABEL_SIZE, ERR_TICK_SIZE)
        X, Y = surface3d(ax, x_vals, y_vals, Z)
        scatter3d(ax, X, Y, series, float(np.nanmin(Z)))
        pads = labelpad3d or AX3D_LABELPAD
        ax.set_xlabel(labels3d[0], labelpad=pads[0])
        ax.set_ylabel(labels3d[1], labelpad=pads[1])
        ax.set_zlabel(labels3d[2], labelpad=pads[2])
        caption(fig, W, H, ax3d_rect[0], base + caption_dy, title)

        for row, (err, name) in enumerate(panels):
            bottom = base + panel_rect[1] + (len(panels) - 1 - row) * panel_pitch
            axe = fig.add_axes(rect(panel_rect[0], bottom,
                                    panel_rect[2], panel_rect[3]))
            caxe = fig.add_axes(rect(panel_rect[0] + panel_rect[2] + CBAR_GAP,
                                     bottom, CBAR_WIDTH, panel_rect[3]))
            style_heatmap_axes(axe, ERR_LABEL_SIZE, ERR_TICK_SIZE, ERR_NBINS)
            cb = heatmap(axe, caxe, x_vals, y_vals, err, reds_dark(),
                         0.0, np.nanmax(err))
            if row == len(panels) - 1:
                axe.set_xlabel(panel_labels[0])
            else:
                axe.set_xticklabels([])
            axe.set_ylabel(panel_labels[1])
            style_colorbar(cb, name + "\nabs. err.", LEGEND_SIZE,
                           ERR_TICK_SIZE, CBAR_ERR_NBINS)

    top_legend(fig, legend_handles, handler_map)
    save(fig, relpath)


def _analytic_handle():
    handle = mpatches.Patch(label="Analytic")
    return handle, {handle: GradientHandler(magma_dark())}


def fig_tfim():
    blocks = []
    for base, folder, title in ((352.704, "obc", "a. Open boundaries"),
                                (36.0, "pbc", "b. Periodic boundaries")):
        analytic = load("tfim", folder, "tfim-1d-%s-E-Lx-vs-h-analytic.json" % folder)
        vqe = load("tfim", folder, "tfim-1d-%s-E-Lx-vs-h-vqe.json" % folder)
        dmrg = load("tfim", folder, "tfim-1d-%s-E-Lx-vs-h-dmrg.json" % folder)
        Lx = np.array(analytic["x_values"], dtype=float)
        h = np.array(analytic["y_values"], dtype=float)
        A = grid(analytic, "analytic")
        V = grid(vqe, "vqe")
        D = grid(dmrg, "dmrg")
        blocks.append((base, title, Lx, h, A,
                       ((D, DMRG_COLOR, "s", DMRG_SIZE),
                        (V, VQE_COLOR, "o", VQE_SIZE)),
                       ((np.abs(V - A), "VQE"), (np.abs(D - A), "DMRG"))))
    handle, hmap = _analytic_handle()
    stacked_figure(
        490.176, 620.448, blocks,
        (10.08, 0.0, 214.464, 214.464), (310.944, 0.0, 102.912, 102.912),
        111.552, 224.069,
        [handle, marker_handle(VQE_COLOR, "o", "VQE"),
         marker_handle(DMRG_COLOR, "s", "DMRG")], hmap,
        "tfim-E-Lx-vs-h.pdf", ("$L_x$", "$h$", "$E$"), ("$L_x$", "$h$"))


def fig_nocc_vs_t2():
    blocks = []
    specs = ((380.592, "nocc-vs-t2", "a. Periodic",
              "simulated-ideal-analytic-E-nocc-vs-t2.json",
              "simulated-ideal-dmrg-E-nocc-vs-t2.json",
              "simulated-ideal-vqe-iqpe-E-nocc-vs-t2.json"),
             (36.0, "nocc-vs-t2-hard-wall", "b. Hard-wall flake",
              "simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json",
              "simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json",
              "simulated-ideal-vqe-iqpe-E-nocc-vs-t2-hard-wall.json"))
    for base, folder, title, a_file, d_file, vi_file in specs:
        analytic = load(folder, a_file)
        dmrg = load(folder, d_file)
        vqe_iqpe = load(folder, vi_file)
        n_occ = np.array(analytic["x_values"], dtype=float)
        t2 = np.array(analytic["y_values"], dtype=float)
        A = grid(analytic, "analytic")
        D = grid(dmrg, "dmrg")
        V = grid(vqe_iqpe, "vqe")
        I = grid(vqe_iqpe, "iqpe")
        blocks.append((base, title, n_occ, t2, A,
                       ((D, DMRG_COLOR, "s", DMRG_SIZE),
                        (V, VQE_COLOR, "o", VQE_SIZE),
                        (I, IQPE_COLOR, "^", IQPE_SIZE)),
                       ((np.abs(V - A), "VQE"), (np.abs(I - A), "IQPE"),
                        (np.abs(D - A), "DMRG"))))
    handle, hmap = _analytic_handle()
    stacked_figure(
        490.176, 676.224, blocks,
        (10.08, 0.0, 242.352, 242.352), (338.832, 0.0, 75.024, 75.024),
        83.664, 251.957,
        [handle, marker_handle(VQE_COLOR, "o", "VQE"),
         marker_handle(IQPE_COLOR, "^", "IQPE"),
         marker_handle(DMRG_COLOR, "s", "DMRG")], hmap,
        "haldane-E-N_occ-vs-t2.pdf",
        (r"$N_\mathrm{occ}$", "$t_2$", "$E$"),
        (r"$N_\mathrm{occ}$", "$t_2$"))


def fig_appendix_m_vs_u():
    blocks = []
    specs = ((333.5483, "M-vs-U", "a. Periodic",
              "simulated-noisy-dmrg-analytic-E-M-vs-U.json"),
             (36.0, "M-vs-U-hard-wall", "b. Hard wall",
              "simulated-noisy-analytic-dmrg-E-M-vs-U-hard-wall.json"))
    for base, folder, title, name in specs:
        data = load("appendix-M-vs-U", folder, name)
        M = np.array(data["x_values"], dtype=float)
        U = np.array(data["y_values"], dtype=float)
        A = grid(data, "analytic")
        D = grid(data, "dmrg")
        blocks.append((base, title, M, U, A,
                       ((D, DMRG_COLOR, "s", DMRG_SIZE),),
                       ((np.abs(D - A), "DMRG"),)))
    handle, hmap = _analytic_handle()
    stacked_figure(
        490.176, 582.1366153846, blocks,
        (10.08, 0.0, 195.3083, 195.3083), (291.7883, 36.6203, 122.06769, 122.06769),
        0.0, 204.9133,
        [handle, marker_handle(DMRG_COLOR, "s", "DMRG")], hmap,
        "haldane-E-M-vs-U.pdf", ("$M$", "$U$", "$E$"), ("$M$", "$U$"))


def fig_boundary_conditions():
    import boundary_conditions

    boundary_conditions.main()


FIGURES = {
    "boundary-conditions": fig_boundary_conditions,
    "M-vs-phi": fig_m_vs_phi,
    "appendix-M-vs-U": fig_appendix_m_vs_u,
    "nocc-vs-t2": fig_nocc_vs_t2,
    "tfim": fig_tfim,
    "band-structure": fig_band_structure,
    "Magnetization": fig_magnetization,
    "hardware-hubbard": fig_hardware_hubbard,
    "hardware-max3sat": fig_hardware_max3sat,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figures", nargs="*", choices=sorted(FIGURES) + [[]],
                    help="subset to regenerate (default: all)")
    args = ap.parse_args()
    if matplotlib.__version__ != MPL_VERSION:
        print("warning: the manuscript figures were rendered with matplotlib "
              "%s, this is %s" % (MPL_VERSION, matplotlib.__version__))
    rcparams()
    for name in (args.figures or sorted(FIGURES)):
        FIGURES[name]()


if __name__ == "__main__":
    main()
