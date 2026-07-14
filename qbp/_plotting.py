from __future__ import annotations

import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

from qbp._interactive import attach_hover, lock_camera_azimuth


def _format_momentum_ticks(ax, axis, vals):
    arr = np.asarray(vals, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    lo_n = int(np.floor(lo / np.pi + 1e-6))
    hi_n = int(np.ceil(hi / np.pi - 1e-6))
    ticks = np.arange(lo_n, hi_n + 1) * np.pi

    def fmt(t):
        if abs(t) < 1e-9:
            return r"$0$"
        if abs(t - np.pi) < 1e-9:
            return r"$\pi$"
        if abs(t + np.pi) < 1e-9:
            return r"$-\pi$"
        n = int(round(t / np.pi))
        return rf"${n}\pi$"

    labels = [fmt(t) for t in ticks]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def _apply_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "figure.dpi": 150,
    })


def _save_and_show(fig, output_path, hide_plot):
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, format=os.path.splitext(output_path)[1].lstrip(".") or "pdf")
        if str(output_path).endswith(".pdf"):
            try:
                subprocess.run(["pdfcrop", output_path, output_path], check=True, capture_output=True)
            except Exception:
                pass
    if not hide_plot:
        plt.show()
    return fig


class _GradientPatchHandler(HandlerBase):
    def __init__(self, cmap):
        self.cmap = cmap
        super().__init__()

    def create_artists(self, _legend, _orig_handle, xdescent, ydescent, width, height, _fontsize, trans):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        line_h = height * 0.18
        line_y = ydescent + (height - line_h) / 2
        bbox = Bbox.from_bounds(xdescent, line_y, width, line_h)
        im = BboxImage(TransformedBbox(bbox, trans), cmap=self.cmap)
        im.set_data(gradient)
        im.set_alpha(0.9)
        dot = Line2D(
            [xdescent + width * 0.5], [ydescent + height * 0.5],
            marker="o", markersize=7, linestyle="none",
            markerfacecolor=self.cmap(0.5), markeredgewidth=0,
        )
        dot.set_transform(trans)
        return [im, dot]


def plot_analytic(
    x_vals,
    y_vals,
    x_label: str,
    y_label: str,
    Z: np.ndarray,
    *,
    plot_format: str = "3d",
    output_path=None,
    hide_plot: bool = False,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
    z_label: str = "$E$",
):
    if plot_format == "2d":
        return _plot_analytic_2d(
            x_vals, x_label, Z,
            x_is_momentum=x_is_momentum,
            output_path=output_path, hide_plot=hide_plot,
            y_label=z_label,
        )
    if plot_format == "heatmap":
        if Z.ndim == 3:
            return _plot_band_structure_heatmap(
                x_vals, y_vals, x_label, y_label, Z,
                vqe=None, iqpe=None,
                x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
                output_path=output_path, hide_plot=hide_plot,
            )
        return plot_analytic_heatmap(
            x_vals, y_vals, x_label, y_label, Z,
            x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
            output_path=output_path, hide_plot=hide_plot,
            z_label=z_label,
        )
    if Z.ndim == 3:
        return _plot_band_structure_3d(
            x_vals, y_vals, x_label, y_label, Z,
            vqe=None, iqpe=None,
            x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
            output_path=output_path, hide_plot=hide_plot,
        )
    return plot_analytic_3d(
        x_vals, y_vals, x_label, y_label, Z,
        x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
        output_path=output_path, hide_plot=hide_plot,
        z_label=z_label,
    )


def plot_analytic_3d(
    x_vals,
    y_vals,
    x_label: str,
    y_label: str,
    Z: np.ndarray,
    *,
    output_path=None,
    hide_plot: bool = False,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
    z_label: str = "$E$",
):
    _apply_rcparams()

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#cccccc")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.view_init(elev=25, azim=-55)
    ax.dist = 7

    cmap_obj = LinearSegmentedColormap.from_list("magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256)))
    ny = len(y_vals)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

    ax.plot_surface(X_grid, Y_grid, Z, cmap=cmap_obj, alpha=0.10, edgecolor="none",
                    rcount=ny, ccount=len(x_vals))
    for iy, yv in enumerate(y_vals):
        color = cmap_obj(iy / max(ny - 1, 1))
        ax.plot(x_vals, [yv] * len(x_vals), Z[:, iy], color=color, linewidth=1.8, alpha=0.95, zorder=5)
        ax.scatter(x_vals, [yv] * len(x_vals), Z[:, iy],
                   color=color, s=20, alpha=0.4, depthshade=False, zorder=5)

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel(z_label, labelpad=10)

    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)
    if y_is_momentum:
        _format_momentum_ticks(ax, "y", y_vals)

    plt.tight_layout()

    attach_hover(fig, ax, [
        {"xs": X_grid.ravel(), "ys": Y_grid.ravel(), "zs": Z.ravel(), "label": "Analytic"},
    ])

    lock_camera_azimuth(fig, ax)
    return _save_and_show(fig, output_path, hide_plot)


def plot_analytic_heatmap(
    x_vals,
    y_vals,
    x_label: str,
    y_label: str,
    Z: np.ndarray,
    *,
    output_path=None,
    hide_plot: bool = False,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
    z_label: str = "$E$",
):
    _apply_rcparams()

    fig, ax = plt.subplots(figsize=(9, 6.5))

    cmap_obj = LinearSegmentedColormap.from_list("magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256)))

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    def _edges(a):
        if len(a) == 1:
            d = 0.5
            return np.array([a[0] - d, a[0] + d])
        d = np.diff(a) / 2.0
        return np.concatenate([[a[0] - d[0]], a[:-1] + d, [a[-1] + d[-1]]])

    x_edges = _edges(x_arr)
    y_edges = _edges(y_arr)

    mesh = ax.pcolormesh(x_edges, y_edges, Z.T, cmap=cmap_obj, shading="auto", rasterized=True)

    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(z_label, labelpad=10)
    cbar.outline.set_edgecolor("#cccccc")

    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)
    if y_is_momentum:
        _format_momentum_ticks(ax, "y", y_vals)

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)


def plot_simulated(
    x_vals,
    y_vals,
    x_label: str,
    y_label: str,
    Z_exact: np.ndarray,
    Z_vqe: np.ndarray | None,
    Z_iqpe: np.ndarray | None,
    *,
    plot_format: str = "3d",
    hide_legend: bool = False,
    output_path=None,
    hide_plot: bool = False,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
    vqe_label: str = "VQE",
    iqpe_label: str = "IQPE",
    surface_label: str = "Analytic",
    extra_series: list[dict] | None = None,
):
    extra_series = extra_series or []
    if plot_format == "2d":
        return _plot_simulated_2d(
            x_vals, x_label, Z_exact, Z_vqe, Z_iqpe,
            x_is_momentum=x_is_momentum,
            hide_legend=hide_legend,
            output_path=output_path, hide_plot=hide_plot,
            vqe_label=vqe_label, iqpe_label=iqpe_label,
            surface_label=surface_label,
            extra_series=extra_series,
        )
    if Z_exact.ndim == 3:
        return _plot_band_structure_3d(
            x_vals, y_vals, x_label, y_label, Z_exact,
            vqe=Z_vqe, iqpe=Z_iqpe,
            x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
            hide_legend=hide_legend,
            output_path=output_path, hide_plot=hide_plot,
            vqe_label=vqe_label, iqpe_label=iqpe_label,
        )
    _apply_rcparams()

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#cccccc")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.view_init(elev=25, azim=-55)
    ax.dist = 7

    cmap_obj = LinearSegmentedColormap.from_list("magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256)))
    ny = len(y_vals)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

    ax.plot_surface(X_grid, Y_grid, Z_exact, cmap=cmap_obj, alpha=0.10, edgecolor="none",
                    rcount=ny, ccount=len(x_vals))
    for iy, yv in enumerate(y_vals):
        color = cmap_obj(iy / max(ny - 1, 1))
        ax.plot(x_vals, [yv] * len(x_vals), Z_exact[:, iy], color=color, linewidth=1.8, alpha=0.9, zorder=4)
        ax.scatter(x_vals, [yv] * len(x_vals), Z_exact[:, iy],
                   color=color, s=20, alpha=0.4, depthshade=False, zorder=5)

    z_clip = float(np.nanmin(Z_exact))
    x_flat, y_flat = X_grid.ravel(), Y_grid.ravel()

    hover_series = [{"xs": X_grid.ravel(), "ys": Y_grid.ravel(), "zs": Z_exact.ravel(), "label": surface_label}]
    legend_handles = [mpatches.Patch(label=surface_label)]
    handler_map = {legend_handles[0]: _GradientPatchHandler(cmap_obj)}

    if Z_vqe is not None:
        vqe_flat = Z_vqe.ravel()
        vqe_mask = vqe_flat >= z_clip
        ax.scatter(x_flat[vqe_mask], y_flat[vqe_mask], vqe_flat[vqe_mask],
                   color="#0072B2", marker="o", s=45, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor="#0072B2", markersize=14, label=vqe_label))
        hover_series.append({"xs": x_flat[vqe_mask], "ys": y_flat[vqe_mask],
                             "zs": vqe_flat[vqe_mask], "label": vqe_label})

    if Z_iqpe is not None:
        iqpe_flat = Z_iqpe.ravel()
        iqpe_mask = iqpe_flat >= z_clip
        ax.scatter(x_flat[iqpe_mask], y_flat[iqpe_mask], iqpe_flat[iqpe_mask],
                   color="#6DBF82", marker="^", s=45, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="^", color="w",
                                     markerfacecolor="#6DBF82", markersize=14, label=iqpe_label))
        hover_series.append({"xs": x_flat[iqpe_mask], "ys": y_flat[iqpe_mask],
                             "zs": iqpe_flat[iqpe_mask], "label": iqpe_label})

    for series in extra_series:
        values = np.asarray(series["values"], dtype=float).ravel()
        mask = values >= z_clip
        color = series.get("color", "#D55E00")
        marker = series.get("marker", "D")
        label = series.get("label", "Series")
        size = series.get("size", 45)
        ax.scatter(x_flat[mask], y_flat[mask], values[mask],
                   color=color, marker=marker, s=size, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker=marker, color="w",
                                     markerfacecolor=color, markersize=14, label=label))
        hover_series.append({"xs": x_flat[mask], "ys": y_flat[mask],
                             "zs": values[mask], "label": label})

    ax.set_zlim(bottom=z_clip)

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel("$E$", labelpad=10)

    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)
    if y_is_momentum:
        _format_momentum_ticks(ax, "y", y_vals)

    if not hide_legend and len(legend_handles) > 1:
        fig.legend(handles=legend_handles, loc="upper center",
                   ncol=len(legend_handles), fontsize=14, bbox_to_anchor=(0.5, 0.98),
                   handler_map=handler_map)

    plt.tight_layout()

    attach_hover(fig, ax, hover_series)

    lock_camera_azimuth(fig, ax)
    return _save_and_show(fig, output_path, hide_plot)


def _plot_band_structure_3d(
    x_vals,
    y_vals,
    x_label: str,
    y_label: str,
    Z_bands: np.ndarray,
    *,
    vqe: np.ndarray | None = None,
    iqpe: np.ndarray | None = None,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
    hide_legend: bool = False,
    output_path=None,
    hide_plot: bool = False,
    vqe_label: str = "VQE",
    iqpe_label: str = "IQPE",
):
    _apply_rcparams()
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#cccccc")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.view_init(elev=20, azim=-55)
    ax.dist = 7

    diverging_cmap = LinearSegmentedColormap.from_list("blue_white_red", ["royalblue", "white", "red"])
    vmin = float(np.nanmin(Z_bands))
    vmax = float(np.nanmax(Z_bands))
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)

    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
    n_bands = Z_bands.shape[-1]
    hover_series = []
    for b in range(n_bands):
        Eb = Z_bands[:, :, b]
        ax.plot_surface(
            X_grid, Y_grid, Eb,
            facecolors=diverging_cmap(norm(Eb)),
            edgecolor="k", linewidth=0.2, antialiased=True,
        )
        hover_series.append({"xs": X_grid.ravel(), "ys": Y_grid.ravel(),
                             "zs": Eb.ravel(), "label": f"Band {b}"})

    legend_handles = []
    if vqe is not None:
        vqe_flat = vqe.ravel()
        ax.scatter(X_grid.ravel(), Y_grid.ravel(), vqe_flat,
                   color="#0072B2", marker="o", s=45, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor="#0072B2", markersize=14, label=vqe_label))
        hover_series.append({"xs": X_grid.ravel(), "ys": Y_grid.ravel(),
                             "zs": vqe_flat, "label": vqe_label})
    if iqpe is not None:
        iqpe_flat = iqpe.ravel()
        ax.scatter(X_grid.ravel(), Y_grid.ravel(), iqpe_flat,
                   color="#6DBF82", marker="^", s=45, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="^", color="w",
                                     markerfacecolor="#6DBF82", markersize=14, label=iqpe_label))
        hover_series.append({"xs": X_grid.ravel(), "ys": Y_grid.ravel(),
                             "zs": iqpe_flat, "label": iqpe_label})

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel("$E(k)$", labelpad=10)

    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)
    if y_is_momentum:
        _format_momentum_ticks(ax, "y", y_vals)

    if not hide_legend and legend_handles:
        fig.legend(handles=legend_handles, loc="upper center",
                   ncol=len(legend_handles), fontsize=14, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout()
    attach_hover(fig, ax, hover_series)
    lock_camera_azimuth(fig, ax)
    return _save_and_show(fig, output_path, hide_plot)


def _plot_band_structure_heatmap(
    x_vals,
    y_vals,
    x_label: str,
    y_label: str,
    Z_bands: np.ndarray,
    *,
    vqe: np.ndarray | None = None,
    iqpe: np.ndarray | None = None,
    x_is_momentum: bool = False,
    y_is_momentum: bool = False,
    output_path=None,
    hide_plot: bool = False,
):
    _apply_rcparams()
    n_bands = Z_bands.shape[-1]
    fig, axes = plt.subplots(1, n_bands, figsize=(7 * n_bands, 6.5))
    if n_bands == 1:
        axes = [axes]

    cmaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    extent = [x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()]

    for b in range(n_bands):
        ax = axes[b]
        Eb = Z_bands[:, :, b]
        c = ax.imshow(Eb.T, origin="lower", cmap=cmaps[b % len(cmaps)],
                      extent=extent, aspect="auto")
        plt.colorbar(c, ax=ax)
        ax.set_title(rf"Band {b}: $E_{b}(k)$")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_is_momentum:
            _format_momentum_ticks(ax, "x", x_vals)
        if y_is_momentum:
            _format_momentum_ticks(ax, "y", y_vals)

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)


def _plot_analytic_2d(
    x_vals,
    x_label: str,
    Z: np.ndarray,
    *,
    x_is_momentum: bool = False,
    output_path=None,
    hide_plot: bool = False,
    y_label: str = "$E$",
):
    _apply_rcparams()
    fig, ax = plt.subplots(figsize=(9, 6))

    x_arr = np.asarray(x_vals, dtype=float)

    if Z.ndim == 2:
        n_bands = Z.shape[-1]
        diverging_cmap = LinearSegmentedColormap.from_list(
            "blue_white_red", ["royalblue", "white", "red"]
        )
        vmin = float(np.nanmin(Z))
        vmax = float(np.nanmax(Z))
        for b in range(n_bands):
            t = (b + 0.5) / n_bands
            color = diverging_cmap(t)
            ax.plot(x_arr, Z[:, b], color=color, linewidth=1.8, alpha=0.95,
                    label=rf"Band {b}", zorder=5)
            ax.scatter(x_arr, Z[:, b], color=color, s=20, alpha=0.4, zorder=6)
        if vmin < 0 < vmax:
            ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
        leg = ax.legend(loc="best", fontsize=10, frameon=True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cccccc")
        leg.get_frame().set_alpha(0.9)
    else:
        cmap_obj = LinearSegmentedColormap.from_list(
            "magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256))
        )
        color = cmap_obj(0.5)
        ax.plot(x_arr, Z, color=color, linewidth=1.8, alpha=0.95, zorder=5)
        ax.scatter(x_arr, Z, color=color, s=20, alpha=0.5, zorder=6)

    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")
    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)


def _plot_simulated_2d(
    x_vals,
    x_label: str,
    Z_exact: np.ndarray,
    Z_vqe: np.ndarray | None,
    Z_iqpe: np.ndarray | None,
    *,
    x_is_momentum: bool = False,
    hide_legend: bool = False,
    output_path=None,
    hide_plot: bool = False,
    vqe_label: str = "VQE",
    iqpe_label: str = "IQPE",
    surface_label: str = "Analytic",
    extra_series: list[dict] | None = None,
):
    extra_series = extra_series or []
    _apply_rcparams()
    fig, ax = plt.subplots(figsize=(9, 6))

    x_arr = np.asarray(x_vals, dtype=float)

    if Z_exact.ndim == 2:
        n_bands = Z_exact.shape[-1]
        diverging_cmap = LinearSegmentedColormap.from_list(
            "blue_white_red", ["royalblue", "white", "red"]
        )
        vmin = float(np.nanmin(Z_exact))
        vmax = float(np.nanmax(Z_exact))
        for b in range(n_bands):
            t = (b + 0.5) / n_bands
            color = diverging_cmap(t)
            ax.plot(x_arr, Z_exact[:, b], color=color, linewidth=1.8, alpha=0.9,
                    label=rf"Analytic Band {b}", zorder=4)
            ax.scatter(x_arr, Z_exact[:, b], color=color, s=18, alpha=0.4, zorder=5)
        if vmin < 0 < vmax:
            ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    else:
        cmap_obj = LinearSegmentedColormap.from_list(
            "magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256))
        )
        color = cmap_obj(0.5)
        ax.plot(x_arr, Z_exact, color=color, linewidth=1.8, alpha=0.9,
                label=surface_label, zorder=4)
        ax.scatter(x_arr, Z_exact, color=color, s=20, alpha=0.5, zorder=5)

    if Z_vqe is not None:
        ax.scatter(x_arr, Z_vqe, color="#0072B2", marker="o", s=45,
                   label=vqe_label, zorder=6)
    if Z_iqpe is not None:
        ax.scatter(x_arr, Z_iqpe, color="#6DBF82", marker="^", s=45,
                   label=iqpe_label, zorder=6)
    for series in extra_series:
        ax.scatter(
            x_arr,
            np.asarray(series["values"], dtype=float),
            color=series.get("color", "#D55E00"),
            marker=series.get("marker", "D"),
            s=series.get("size", 45),
            label=series.get("label", "Series"),
            zorder=6,
        )

    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel("$E$", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")
    if x_is_momentum:
        _format_momentum_ticks(ax, "x", x_vals)

    if not hide_legend:
        leg = ax.legend(loc="best", fontsize=11, frameon=True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cccccc")
        leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)
