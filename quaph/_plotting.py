from __future__ import annotations

import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

from quaph._interactive import attach_hover, lock_camera_azimuth


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
    output_path=None,
    hide_plot: bool = False,
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
    ax.set_zlabel("$E$", labelpad=10)

    plt.tight_layout()

    attach_hover(fig, ax, [
        {"xs": X_grid.ravel(), "ys": Y_grid.ravel(), "zs": Z.ravel(), "label": "Analytic"},
    ])

    lock_camera_azimuth(fig, ax)
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
    hide_legend: bool = False,
    output_path=None,
    hide_plot: bool = False,
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

    ax.plot_surface(X_grid, Y_grid, Z_exact, cmap=cmap_obj, alpha=0.10, edgecolor="none",
                    rcount=ny, ccount=len(x_vals))
    for iy, yv in enumerate(y_vals):
        color = cmap_obj(iy / max(ny - 1, 1))
        ax.plot(x_vals, [yv] * len(x_vals), Z_exact[:, iy], color=color, linewidth=1.8, alpha=0.9, zorder=4)
        ax.scatter(x_vals, [yv] * len(x_vals), Z_exact[:, iy],
                   color=color, s=20, alpha=0.4, depthshade=False, zorder=5)

    z_clip = float(np.nanmin(Z_exact))
    x_flat, y_flat = X_grid.ravel(), Y_grid.ravel()

    hover_series = [{"xs": X_grid.ravel(), "ys": Y_grid.ravel(), "zs": Z_exact.ravel(), "label": "Analytic"}]
    legend_handles = [mpatches.Patch(label="Analytic")]
    handler_map = {legend_handles[0]: _GradientPatchHandler(cmap_obj)}

    if Z_vqe is not None:
        vqe_flat = Z_vqe.ravel()
        vqe_mask = vqe_flat >= z_clip
        ax.scatter(x_flat[vqe_mask], y_flat[vqe_mask], vqe_flat[vqe_mask],
                   color="#0072B2", marker="o", s=45, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor="#0072B2", markersize=14, label="VQE"))
        hover_series.append({"xs": x_flat[vqe_mask], "ys": y_flat[vqe_mask],
                             "zs": vqe_flat[vqe_mask], "label": "VQE"})

    if Z_iqpe is not None:
        iqpe_flat = Z_iqpe.ravel()
        iqpe_mask = iqpe_flat >= z_clip
        ax.scatter(x_flat[iqpe_mask], y_flat[iqpe_mask], iqpe_flat[iqpe_mask],
                   color="#6DBF82", marker="^", s=45, depthshade=True, zorder=6)
        legend_handles.append(Line2D([0], [0], marker="^", color="w",
                                     markerfacecolor="#6DBF82", markersize=14, label="IQPE"))
        hover_series.append({"xs": x_flat[iqpe_mask], "ys": y_flat[iqpe_mask],
                             "zs": iqpe_flat[iqpe_mask], "label": "IQPE"})

    ax.set_zlim(bottom=z_clip)

    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel("$E$", labelpad=10)

    if not hide_legend and len(legend_handles) > 1:
        fig.legend(handles=legend_handles, loc="upper center",
                   ncol=len(legend_handles), fontsize=14, bbox_to_anchor=(0.5, 0.98),
                   handler_map=handler_map)

    plt.tight_layout()

    attach_hover(fig, ax, hover_series)

    lock_camera_azimuth(fig, ax)
    return _save_and_show(fig, output_path, hide_plot)
