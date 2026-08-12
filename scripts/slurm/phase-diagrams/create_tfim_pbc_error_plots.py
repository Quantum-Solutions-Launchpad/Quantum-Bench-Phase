#!/usr/bin/env python3
import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from qbp._plotting import _apply_rcparams, _save_and_show
from qbp._diff import _pcolormesh_edges

# Data directory
eval_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/tfim/pbc"
output_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/tfim/pbc"

# Load VQE data
with open(f"{eval_dir}/tfim-1d-pbc-E-Lx-vs-h-vqe.json") as f:
    vqe_data = json.load(f)

# Load analytic data
with open(f"{eval_dir}/tfim-1d-pbc-E-Lx-vs-h-analytic.json") as f:
    analytic_data = json.load(f)

x_values = np.array(vqe_data["x_values"])
y_values = np.array(vqe_data["y_values"])
nx, ny = len(x_values), len(y_values)

def extract_grid_from_result(data, method_name):
    """Extract 2D grid from result data (dense format)"""
    result = data["result"][method_name]
    grid = np.full((nx, ny), np.nan)

    for xi_str, yi_dict in result.items():
        xi = int(xi_str)
        for yi_str, value in yi_dict.items():
            yi = int(yi_str)
            if xi < nx and yi < ny:
                grid[xi, yi] = value

    return grid

# Extract grids
Z_vqe = extract_grid_from_result(vqe_data, "vqe")
Z_analytic = extract_grid_from_result(analytic_data, "analytic")

# Compute errors
absolute_error = Z_vqe - Z_analytic
relative_error = np.abs(absolute_error) / (np.abs(Z_analytic) + 1e-10)

print(f"TFIM PBC - Grid shape: {nx} x {ny}")
print(f"Mean absolute error: {np.nanmean(np.abs(absolute_error)):.6f}")
print(f"Max absolute error: {np.nanmax(np.abs(absolute_error)):.6f}")
print(f"Mean relative error: {np.nanmean(relative_error[np.isfinite(relative_error)]):.6f}")

_apply_rcparams()

def create_error_heatmap(x_vals, y_vals, Z_err, z_label, output_path):
    """Create error heatmap using magma colormap"""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    cmap = plt.cm.magma
    norm = Normalize(vmin=np.nanmin(Z_err), vmax=np.nanmax(Z_err))

    mesh = ax.pcolormesh(
        _pcolormesh_edges(x_arr), _pcolormesh_edges(y_arr), Z_err.T,
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(z_label, labelpad=10)
    cbar.outline.set_edgecolor("#cccccc")

    ax.set_xlabel(r"$L_x$", labelpad=8)
    ax.set_ylabel(r"$h$", labelpad=8)
    ax.set_xlim(_pcolormesh_edges(x_arr)[0], _pcolormesh_edges(x_arr)[-1])
    ax.set_ylim(_pcolormesh_edges(y_arr)[0], _pcolormesh_edges(y_arr)[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")

    plt.tight_layout()
    _save_and_show(fig, output_path, hide_plot=True)

# 1. Absolute Error Heatmap
create_error_heatmap(
    x_values, y_values, absolute_error,
    z_label="Absolute Error",
    output_path=f"{output_dir}/error-abs-vqe-E-Lx-vs-h.pdf",
)
print(f"Saved: {output_dir}/error-abs-vqe-E-Lx-vs-h.pdf")

# 2. Relative Error Heatmap (in percentages)
create_error_heatmap(
    x_values, y_values, (relative_error * 100),
    z_label="Relative Error (%)",
    output_path=f"{output_dir}/error-rel-vqe-E-Lx-vs-h.pdf",
)
print(f"Saved: {output_dir}/error-rel-vqe-E-Lx-vs-h.pdf")

print("TFIM PBC error plots created successfully!")
