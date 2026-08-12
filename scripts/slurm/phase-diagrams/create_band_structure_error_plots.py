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
eval_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/band-structure"
output_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/band-structure"

# Load VQE data
with open(f"{eval_dir}/simulated-noisy-vqe-E-kx-vs-ky.json") as f:
    vqe_data = json.load(f)

# Load analytic data
with open(f"{eval_dir}/simulated-noisy-analytic-E-kx-vs-ky.json") as f:
    analytic_data = json.load(f)

x_values = np.array(vqe_data["x_values"])
y_values = np.array(vqe_data["y_values"])

print(f"Grid dimensions: {len(x_values)} x {len(y_values)}")

# Extract VQE data - band structure format (17 bands x 17 k-points)
vqe_result = vqe_data["result"]["vqe"]
analytic_result = analytic_data["result"]["analytic"]

n_bands = len(vqe_result)
n_kpoints = len(vqe_result["0"])

# Create 2D grids for band structure (treating bands as x-axis, k-points as y-axis)
Z_vqe = np.zeros((n_bands, n_kpoints))
Z_analytic = np.zeros((n_bands, n_kpoints))

# Extract data
for band_idx in range(n_bands):
    band_data_vqe = vqe_result[str(band_idx)]
    band_data_analytic = analytic_result[str(band_idx)]
    for k_idx in range(n_kpoints):
        Z_vqe[band_idx, k_idx] = band_data_vqe[str(k_idx)]
        val = band_data_analytic[str(k_idx)]
        # Handle both scalar and vector formats (vector = [energy, phase])
        if isinstance(val, (list, tuple)):
            Z_analytic[band_idx, k_idx] = val[0]
        else:
            Z_analytic[band_idx, k_idx] = val

# Compute errors
absolute_error = Z_vqe - Z_analytic
relative_error = np.abs(absolute_error) / (np.abs(Z_analytic) + 1e-10)

print(f"VQE data shape: {Z_vqe.shape}")
print(f"Mean absolute error: {np.nanmean(np.abs(absolute_error)):.6e}")
print(f"Max absolute error: {np.nanmax(np.abs(absolute_error)):.6e}")
print(f"Mean relative error: {np.nanmean(relative_error[np.isfinite(relative_error)]):.6f}")

_apply_rcparams()

def create_error_heatmap(x_vals, y_vals, Z_err, z_label, output_path):
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

    ax.set_xlabel(r"$k_x$", labelpad=8)
    ax.set_ylabel(r"$k_y$", labelpad=8)
    ax.set_xlim(_pcolormesh_edges(x_arr)[0], _pcolormesh_edges(x_arr)[-1])
    ax.set_ylim(_pcolormesh_edges(y_arr)[0], _pcolormesh_edges(y_arr)[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")

    plt.tight_layout()
    _save_and_show(fig, output_path, hide_plot=True)

# 1. Absolute Error Heatmap
create_error_heatmap(
    x_values, y_values, absolute_error.T,
    z_label="Absolute Error",
    output_path=f"{output_dir}/error-abs-vqe-E-kx-vs-ky.pdf",
)
print(f"Saved: {output_dir}/error-abs-vqe-E-kx-vs-ky.pdf")

# 2. Relative Error Heatmap (in percentages)
create_error_heatmap(
    x_values, y_values, (relative_error * 100).T,
    z_label="Relative Error (%)",
    output_path=f"{output_dir}/error-rel-vqe-E-kx-vs-ky.pdf",
)
print(f"Saved: {output_dir}/error-rel-vqe-E-kx-vs-ky.pdf")

print("\nBand structure error plots created successfully!")
