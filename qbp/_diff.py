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
    """Plot method-vs-method energy differences for a run log.

    Reads a JSON run log written by :func:`~qbp.run` at ``path`` and draws the
    signed energy difference for every pair of methods present in the log,
    delegating to the same implementation as the run/plot ``--diff`` pipeline.
    ``method`` is ``"vqe"``, ``"iqpe"``, or ``"both"``; anything other than
    ``"both"`` restricts the output to pairs involving that method.
    ``plot_format`` is ``"3d"``, ``"heatmap"``, or ``"bar_2d"``. Returns the
    Matplotlib figure(s).
    """
    from qbp._run import load_result, _plot_diffs, _result_labels


    _apply_rcparams()
    rr = load_result(path)
    x_label, y_label, xm, ym = _result_labels(rr.model_name, rr.x_param, rr.y_param)
    selected = None if method == "both" else [method]
    figs = _plot_diffs(
        rr, x_label=x_label, y_label=y_label,
        x_is_momentum=x_is_momentum or xm, y_is_momentum=y_is_momentum or ym,
        plot_format=plot_format, output_path=output_path, hide_plot=hide_plot,
        methods=selected,
    )
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
    *, x_label, y_label, z_label, hover_label, title,
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
    ax.set_title(title, pad=14)

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
    *, x_label, y_label, z_label, title,
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
    ax.set_title(title, pad=10)
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
    *, x_label, y_label, z_label, title,
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
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcece")
    ax.tick_params(direction="out", length=4, color="#888888")

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)