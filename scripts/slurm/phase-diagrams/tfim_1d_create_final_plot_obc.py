#!/usr/bin/env python3
import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')
# source /pscratch/sd/m/mbao202/NNL-P7/venv/bin/activate && python /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/tfim_1d_create_final_plot_obc.py

from qbp._plotting import plot_simulated

# Directories
data_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/logs/tfim/1d/ground-state-energy/Lx-vs-h-obc"
output_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/tfim/1d/ground-state-energy/Lx-vs-h-obc"

# Load analytical and VQE+IQPE data
with open(f"{data_dir}/tfim-1d-obc-E-Lx-vs-h-analytic.json") as f:
    analytic_data = json.load(f)

with open(f"{data_dir}/tfim-1d-obc-E-Lx-vs-h-vqe-iqpe.json") as f:
    vqe_iqpe_data = json.load(f)

x_values = analytic_data["x_values"]
y_values = analytic_data["y_values"]
nx, ny = len(x_values), len(y_values)

def extract_grid(data, method_name):
    """Extract 2D grid from result data"""
    result = data["result"][method_name]
    grid = np.full((nx, ny), np.nan)

    for xi_idx in range(nx):
        for yi_idx in range(ny):
            if str(xi_idx) in result and str(yi_idx) in result[str(xi_idx)]:
                grid[xi_idx, yi_idx] = result[str(xi_idx)][str(yi_idx)]

    return grid

# Extract grids
Z_analytic = extract_grid(analytic_data, "analytic")
Z_vqe = extract_grid(vqe_iqpe_data, "vqe")
Z_iqpe = extract_grid(vqe_iqpe_data, "iqpe")

print(f"Analytical shape: {Z_analytic.shape}, NaN count: {np.isnan(Z_analytic).sum()}")
print(f"VQE shape: {Z_vqe.shape}, NaN count: {np.isnan(Z_vqe).sum()}")
print(f"IQPE shape: {Z_iqpe.shape}, NaN count: {np.isnan(Z_iqpe).sum()}")

# Create combined plot with VQE/IQPE and Analytic
plot_simulated(
    x_values,
    y_values,
    x_label=r"$L_x$",
    y_label=r"$h$",
    Z_exact=Z_analytic,
    Z_vqe=Z_vqe,
    Z_iqpe=Z_iqpe,
    plot_format="3d",
    vqe_label="VQE",
    iqpe_label="IQPE",
    surface_label="Analytic",
    output_path=f"{output_dir}/tfim-1d-obc-E-Lx-vs-h-combined.pdf",
    hide_plot=False,
)

print(f"Plot saved to {output_dir}/tfim-1d-obc-E-Lx-vs-h-combined.pdf")
