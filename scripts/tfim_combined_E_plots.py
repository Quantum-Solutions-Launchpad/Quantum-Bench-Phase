#!/usr/bin/env python3
"""Generate combined E plots for TFIM OBC and PBC using qbp plotting."""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

import matplotlib.pyplot as plt
import numpy as np
from qbp._plotting import _apply_rcparams, _save_and_show
from qbp._diff import _pcolormesh_edges

# Apply QBP style
_apply_rcparams()

eval_base = "/pscratch/sd/m/mbao202/NNL-P7/evaluation"
output_base = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/new-data/tfim"
Path(output_base).mkdir(parents=True, exist_ok=True)

# Load TFIM data for OBC and PBC
for variant in ["obc", "pbc"]:
    eval_dir = f"{eval_base}/tfim/{variant}"

    print(f"Generating E plot for TFIM {variant.upper()}...")

    try:
        # Load analytical data (reference)
        with open(f"{eval_dir}/tfim-1d-{variant}-E-Lx-vs-h-analytic.json") as f:
            analytic_data = json.load(f)

        # Load VQE data
        with open(f"{eval_dir}/tfim-1d-{variant}-E-Lx-vs-h-vqe.json") as f:
            vqe_data = json.load(f)

        # Load DMRG data
        with open(f"{eval_dir}/tfim-1d-{variant}-E-Lx-vs-h-dmrg.json") as f:
            dmrg_data = json.load(f)

        x_values = np.array(analytic_data["x_values"])
        y_values = np.array(analytic_data["y_values"])
        nx, ny = len(x_values), len(y_values)

        # Extract grids from results
        def extract_grid(data, method_name):
            result = data["result"][method_name]
            grid = np.full((nx, ny), np.nan)
            for xi_str, yi_dict in result.items():
                xi = int(xi_str)
                for yi_str, value in yi_dict.items():
                    yi = int(yi_str)
                    if xi < nx and yi < ny:
                        grid[xi, yi] = value
            return grid

        Z_analytic = extract_grid(analytic_data, "analytic")
        Z_vqe = extract_grid(vqe_data, "vqe")
        Z_dmrg = extract_grid(dmrg_data, "dmrg")

        # Create figure with 3 subplots (Analytic, VQE, DMRG)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Colormap and normalization
        vmin = min(np.nanmin(Z_analytic), np.nanmin(Z_vqe), np.nanmin(Z_dmrg))
        vmax = max(np.nanmax(Z_analytic), np.nanmax(Z_vqe), np.nanmax(Z_dmrg))
        cmap = plt.cm.viridis

        # Plot Analytic
        mesh1 = axes[0].pcolormesh(
            _pcolormesh_edges(x_values), _pcolormesh_edges(y_values), Z_analytic.T,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True,
        )
        axes[0].set_xlabel(r"$L_x$", fontsize=24)
        axes[0].set_ylabel(r"$h$", fontsize=24)
        axes[0].set_title("Analytic", fontsize=24)
        axes[0].tick_params(labelsize=18)
        for spine in axes[0].spines.values():
            spine.set_edgecolor("#cccccc")

        # Plot VQE
        mesh2 = axes[1].pcolormesh(
            _pcolormesh_edges(x_values), _pcolormesh_edges(y_values), Z_vqe.T,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True,
        )
        axes[1].set_xlabel(r"$L_x$", fontsize=24)
        axes[1].set_ylabel(r"$h$", fontsize=24)
        axes[1].set_title("VQE", fontsize=24)
        axes[1].tick_params(labelsize=18)
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#cccccc")

        # Plot DMRG
        mesh3 = axes[2].pcolormesh(
            _pcolormesh_edges(x_values), _pcolormesh_edges(y_values), Z_dmrg.T,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True,
        )
        axes[2].set_xlabel(r"$L_x$", fontsize=24)
        axes[2].set_ylabel(r"$h$", fontsize=24)
        axes[2].set_title("DMRG", fontsize=24)
        axes[2].tick_params(labelsize=18)
        for spine in axes[2].spines.values():
            spine.set_edgecolor("#cccccc")

        # Add shared colorbar
        cbar = fig.colorbar(mesh3, ax=axes, pad=0.02, fraction=0.045)
        cbar.set_label("Energy", fontsize=20)
        cbar.ax.tick_params(labelsize=18)

        plt.tight_layout()

        output_path = f"{output_base}/E-Lx-vs-h-{variant}.pdf"
        _save_and_show(fig, output_path, hide_plot=True)
        print(f"✓ Saved: {output_path}")

    except Exception as e:
        print(f"✗ Error for TFIM {variant.upper()}: {e}")
        import traceback
        traceback.print_exc()

print("\n✓ All TFIM E plots generated!")
