from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from qbp._plotting import (
    _apply_rcparams,
    _save_and_show,
    _format_momentum_ticks,
)
from qbp._interactive import attach_hover, lock_camera_azimuth
import json

def _load_grid(path: str, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path) as fh:
        raw = json.load(fh)

    x_vals = np.asarray(raw["x_values"], dtype=float)
    y_vals = np.asarray(raw["y_values"], dtype=float)
    nx, ny = len(x_vals), len(y_vals)

    block = raw["result"][source]
    analytic_block = raw["result"].get("analytic")

    Z = np.full((nx, ny), np.nan)
    for xi_str, row in block.items():
        xi = int(xi_str)
        for yi_str, val in row.items():
            yi = int(yi_str)
            if source in ("vqe", "iqpe") and isinstance(val, list) and analytic_block is not None:
                analytic_val = float(analytic_block[xi_str][yi_str])
                Z[xi, yi] = min(val, key=lambda e: abs(e - analytic_val))
            else:
                # Band structure: val may be a list of eigenvalues — take ground state
                if isinstance(val, list):
                    Z[xi, yi] = float(min(val))
                else:
                    Z[xi, yi] = float(val)

    return x_vals, y_vals, Z


def _read_meta(path: str) -> dict:
    with open(path) as fh:
        raw = json.load(fh)
    return {
        "model": raw.get("parameters", {}).get("model", ""),
        "lattice": raw.get("parameters", {}).get("lattice", []),
        "x_param": raw.get("x_param", "x"),
        "y_param": raw.get("y_param", "y"),
        "available": list(raw.get("result", {}).keys()),
    }


def _param_label(param: str) -> str:
    label_map = {
        "n_occ": r"$N_{\mathrm{occ}}$",
        "t2": r"$t_2$",
        "t1": r"$t_1$",
        "phi": r"$\phi$",
        "M": r"$M$",
        "U": r"$U$",
        "kx": r"$k_x$",
        "ky": r"$k_y$",
        "k": r"$k$",
    }
    return label_map.get(param, f"${param}$")


def _method_name(method: str) -> str:
    return {"vqe": "VQE", "iqpe": "IQPE"}.get(method, method.upper())

def _make_title(method: str, meta: dict) -> str:
    method_name = _method_name(method)
    model = meta.get("model", "")
    lattice = meta.get("lattice", [])

    details = []
    if model:
        details.append(str(model))
    if lattice:
        details.append("×".join(str(v) for v in lattice))

    suffix = f" ({', '.join(details)})" if details else ""
    return f"{method_name} Error Relative to Analytic{suffix}"


def _out_path(base: str | None, method: str) -> str | None:
    if base is None:
        return None
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        return f"{stem}-{method}.{ext}"
    return f"{base}-{method}"

def plot_diff(
    path: str,
    *,
    method: str = "both",
    plot_format: str = "3d",
    output_path: str | None = None,
    hide_plot: bool = False,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
):
    """Plot a quantum method's error relative to the analytic surface.

    Reads a JSON run log written by :func:`~qbp.run` at ``path`` and draws the
    signed difference ``E_method - E_analytic`` over the swept parameters.
    ``method`` is ``"vqe"``, ``"iqpe"``, or ``"both"``; ``plot_format`` is
    ``"3d"``, ``"heatmap"``, or ``"bar_2d"``. Both methods must have been run
    alongside ``Method.ANALYTIC`` for the difference to exist. Returns the
    Matplotlib figure(s).
    """
    _apply_rcparams()

    meta = _read_meta(path)
    x_label = _param_label(meta["x_param"])
    y_label = _param_label(meta["y_param"])

    x_vals, y_vals, Z_analytic = _load_grid(path, "analytic")

    methods = ["vqe", "iqpe"] if method == "both" else [method]
    methods = [m for m in methods if m in meta["available"]]

    figs = []
    for m in methods:
        x_vals, y_vals, Z_method = _load_grid(path, m)
        Z_err = Z_method - Z_analytic

        shared = dict(
            x_label=x_label, y_label=y_label,
            z_label=rf"$E_{{\mathrm{{{m.upper()}}}}} - E_{{\mathrm{{analytic}}}}$",
            method=m, meta=meta,
            x_is_momentum=x_is_momentum,
            y_is_momentum=y_is_momentum,
            output_path=_out_path(output_path, m),
            hide_plot=hide_plot,
        )

        if plot_format == "heatmap":
            fig = _diff_heatmap(x_vals, y_vals, Z_err, **shared)
        elif plot_format == "bar_2d":
            fig = _diff_bar_2d(x_vals, y_vals, Z_err, **shared)
        else:
            fig = _diff_3d(x_vals, y_vals, Z_err, hover_label=f"{_method_name(m)} error", **shared)


    return figs if len(figs) > 1 else figs[0] if figs else None

def _diff_cmap_and_norm(Z: np.ndarray):
    finite = Z[np.isfinite(Z)]
    vmin = float(finite.min()) if len(finite) else -1.0
    vmax = float(finite.max()) if len(finite) else 1.0

    if vmin >= 0.0:
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-12))
        cmap = LinearSegmentedColormap.from_list(
            "diff_wr", ["#FFF5F0", "#D6604D", "#67001F"]
        )
    elif vmax <= 0.0:
        norm = Normalize(vmin=min(vmin, -1e-12), vmax=0.0)
        cmap = LinearSegmentedColormap.from_list(
            "diff_wb", ["#053061", "#2166AC", "#F7F7F7"]
        )
    else:
        abs_max = max(abs(vmin), abs(vmax), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        cmap = LinearSegmentedColormap.from_list(
            "diff_bwr", ["#2166AC", "#F7F7F7", "#D6604D"]
        )
    return cmap, norm


def _pcolormesh_edges(a: np.ndarray) -> np.ndarray:
    if len(a) == 1:
        return np.array([a[0] - 0.5, a[0] + 0.5])
    d = np.diff(a) / 2.0
    return np.concatenate([[a[0] - d[0]], a[:-1] + d, [a[-1] + d[-1]]])


def _diff_3d(
    x_vals, y_vals, Z_err,
    *, x_label, y_label, z_label, hover_label,
    method, meta,
    x_is_momentum, y_is_momentum,
    output_path, hide_plot,
):
    cmap, norm = _diff_cmap_and_norm(Z_err)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#cccccc")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.view_init(elev=25, azim=-55)

    ny = len(y_vals)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

    # I think can remove this if we use a good colormap and alpha, but I think it helps visually to have a faint surface showing the overall error landscape
    ax.plot_surface(
        X_grid, Y_grid, Z_err,
        cmap=cmap, norm=norm, alpha=0.10, edgecolor="none",
        rcount=ny, ccount=len(x_vals),
    )

    # I used one colored line + scatter per y-slice so it matches the other 3D style that we hav for other plots
    for iy, yv in enumerate(y_vals):
        color = cmap(norm(float(np.nanmean(Z_err[:, iy]))))
        ax.plot(x_vals, [yv] * len(x_vals), Z_err[:, iy],
                color=color, linewidth=1.8, alpha=0.95, zorder=5)
        ax.scatter(x_vals, [yv] * len(x_vals), Z_err[:, iy],
                   color=color, s=20, alpha=0.4, depthshade=False, zorder=5)

    ax.plot_surface(
        X_grid, Y_grid, np.zeros_like(Z_err),
        color="#888888", alpha=0.08, edgecolor="none",
    )

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(_make_title(method, meta), pad=14)

    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)
    if y_is_momentum:
        _format_momentum_ticks(ax, "y", y_vals)

    plt.tight_layout()
    attach_hover(fig, ax, [
        {"xs": X_grid.ravel(), "ys": Y_grid.ravel(),
         "zs": Z_err.ravel(), "label": hover_label},
    ])
    lock_camera_azimuth(fig, ax)
    return _save_and_show(fig, output_path, hide_plot)

def _diff_heatmap(
    x_vals, y_vals, Z_err,
    *, x_label, y_label, z_label,
    method, meta,
    x_is_momentum, y_is_momentum,
    output_path, hide_plot,
):
    cmap, norm = _diff_cmap_and_norm(Z_err)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    mesh = ax.pcolormesh(
        _pcolormesh_edges(x_arr), _pcolormesh_edges(y_arr), Z_err.T,
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(z_label, labelpad=10)
    cbar.outline.set_edgecolor("#cccccc")

    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_title(_make_title(method, meta), pad=10)
    ax.set_xlim(_pcolormesh_edges(x_arr)[0], _pcolormesh_edges(x_arr)[-1])
    ax.set_ylim(_pcolormesh_edges(y_arr)[0], _pcolormesh_edges(y_arr)[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")

    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)
    if y_is_momentum:
        _format_momentum_ticks(ax, "y", y_vals)

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)

def _diff_bar_2d(
    x_vals, y_vals, Z_err,
    *, x_label, y_label, z_label,
    method, meta,
    x_is_momentum, y_is_momentum,
    output_path, hide_plot,
):
    nx, ny = Z_err.shape
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    pos_cmap = LinearSegmentedColormap.from_list("pos", ["#FCBBA1", "#D6604D", "#67001F"])
    neg_cmap = LinearSegmentedColormap.from_list("neg", ["#C6DBEF", "#2166AC", "#053061"])

    fig, ax = plt.subplots(figsize=(max(9, nx * 0.7), 6))

    bar_width = 0.8 / max(ny, 1)
    offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, ny)

    for iy in range(ny):
        t = iy / max(ny - 1, 1)
        for ix in range(nx):
            val = Z_err[ix, iy]
            if not np.isfinite(val):
                continue
            color = pos_cmap(t) if val >= 0 else neg_cmap(t)
            ax.bar(ix + offsets[iy], val, width=bar_width * 0.9,
                   color=color, alpha=0.85, zorder=4)

    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7, zorder=3)

    ax.set_xticks(np.arange(nx))
    if x_is_momentum:
        pi_labels = []
        for v in x_arr:
            n = v / np.pi
            if abs(n) < 1e-9:
                pi_labels.append(r"$0$")
            elif abs(abs(n) - 1) < 1e-9:
                pi_labels.append(r"$\pi$" if n > 0 else r"$-\pi$")
            else:
                pi_labels.append(rf"${n:.2g}\pi$")
        ax.set_xticklabels(pi_labels, fontsize=9)
    else:
        ax.set_xticklabels([f"{v:.3g}" for v in x_arr], fontsize=9,
                           rotation=45, ha="right")

    handles = []
    mid_cmap = LinearSegmentedColormap.from_list("mid", ["#D6604D", "#2166AC"])
    for iy in range(ny):
        t = iy / max(ny - 1, 1)
        yv = y_arr[iy]
        if y_is_momentum:
            n = yv / np.pi
            if abs(n) < 1e-9:
                ylabel_str = r"$0$"
            elif abs(abs(n) - 1) < 1e-9:
                ylabel_str = r"$\pi$" if n > 0 else r"$-\pi$"
            else:
                ylabel_str = rf"${n:.2g}\pi$"
        else:
            ylabel_str = f"{yv:.3g}"
        c = mid_cmap(t)
        handles.append(Patch(facecolor=c, alpha=0.85,
                             label=f"{y_label} = {ylabel_str}"))
    if ny <= 12:
        ax.legend(handles=handles, loc="best", fontsize=9,
                  frameon=True, framealpha=0.9, edgecolor="#cccccc")

    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(z_label, labelpad=8)
    ax.set_title(_make_title(method, meta), pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcece")
    ax.tick_params(direction="out", length=4, color="#888888")

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)