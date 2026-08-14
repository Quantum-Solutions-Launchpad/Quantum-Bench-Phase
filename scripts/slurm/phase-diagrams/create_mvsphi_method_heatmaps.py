#!/usr/bin/env python3
import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from qbp._plotting import _apply_rcparams, _save_and_show, _format_momentum_ticks

# Data directory
eval_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/M-vs-phi"
output_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/M-vs-phi"

methods = ["analytic", "dmrg", "vqe"]

def load_method_data(method_name):
    """Load data for a specific method"""
    path = f"{eval_dir}/simulated-ideal-{method_name}-E-M-vs-phi.json"
    with open(path) as f:
        return json.load(f)

def extract_grid_from_result(data, method_name, nx, ny):
    """Extract 2D grid from result data"""
    result = data["result"][method_name]
    grid = np.full((nx, ny), np.nan)

    for xi_str, yi_dict in result.items():
        xi = int(xi_str)
        for yi_str, value in yi_dict.items():
            yi = int(yi_str)
            if xi < nx and yi < ny:
                grid[xi, yi] = value

    return grid

# Load first dataset to get coordinate values
first_data = load_method_data(methods[0])
x_values = np.array(first_data["x_values"])
y_values = np.array(first_data["y_values"])
nx, ny = len(x_values), len(y_values)

_apply_rcparams()

def _edges(a):
    """Compute bin edges for pcolormesh"""
    if len(a) == 1:
        d = 0.5
        return np.array([a[0] - d, a[0] + d])
    d = np.diff(a) / 2.0
    return np.concatenate([[a[0] - d[0]], a[:-1] + d, [a[-1] + d[-1]]])

def create_method_heatmap(x_vals, y_vals, Z, method_name, output_path):
    """Create heatmap for a single method matching standard format"""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Use modified magma colormap like the standard heatmaps
    cmap_obj = LinearSegmentedColormap.from_list("magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256)))

    x_edges = _edges(x_arr)
    y_edges = _edges(y_arr)

    mesh = ax.pcolormesh(x_edges, y_edges, Z.T, cmap=cmap_obj, shading="auto", rasterized=True)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("Energy", labelpad=10)
    cbar.outline.set_edgecolor("#cccccc")

    # Format method name for title
    title_name = method_name.upper() if method_name in ["vqe", "dmrg"] else method_name.capitalize()
    ax.set_title(title_name, pad=10)
    ax.set_xlabel(r"$\phi$", labelpad=8)
    ax.set_ylabel(r"$M$", labelpad=8)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    # Set x-axis ticks from -3 to 3 in steps of 1
    ax.set_xticks(np.arange(-3, 4, 1))

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")

    plt.tight_layout()
    _save_and_show(fig, output_path, hide_plot=True)

# Generate heatmaps for each method
print("Generating heatmaps for each method...")
for method_name in methods:
    print(f"\nProcessing {method_name.upper()}...")

    # Load data
    data = load_method_data(method_name)
    Z = extract_grid_from_result(data, method_name, nx, ny)

    # Compute statistics
    finite_vals = Z[np.isfinite(Z)]
    if len(finite_vals) > 0:
        print(f"  Min value: {np.nanmin(Z):.6f}")
        print(f"  Max value: {np.nanmax(Z):.6f}")
        print(f"  Mean value: {np.nanmean(Z):.6f}")
        print(f"  Missing values: {np.isnan(Z).sum()} / {nx*ny}")

    # Create heatmap
    output_path = f"{output_dir}/{method_name}-heatmap-E-phi-vs-M.pdf"
    create_method_heatmap(x_values, y_values, Z, method_name, output_path)
    print(f"  Saved: {output_path}")

print("\nM-vs-phi method heatmaps created successfully!")
