#!/usr/bin/env python3
import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')
# source /pscratch/sd/m/mbao202/NNL-P7/venv/bin/activate && python /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/M_vs_phi_create_final_plot.py

from qbp._plotting import plot_analytic_heatmap

# Data directory
data_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/logs/haldane/2x2/ground-state-energy/M-vs-phi"
old_data_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/logs/old-haldane-data"
output_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/haldane/2x2/ground-state-energy/M-vs-phi"

# Load VQE+IQPE data
with open(f"{data_dir}/simulated-ideal-vqe-iqpe-E-M-vs-phi.raw-data.json") as f:
    vqe_iqpe_data = json.load(f)

# Load DMRG data from old-haldane-data
with open(f"{old_data_dir}/simulated-ideal-dmrg-E-M-vs-phi.json") as f:
    dmrg_data = json.load(f)

x_values = vqe_iqpe_data["x_values"]
y_values = vqe_iqpe_data["y_values"]
nx, ny = len(x_values), len(y_values)

def extract_grid(data, method_name):
    """Extract 2D grid from cells data"""
    cells = data["cells"][method_name]
    grid = np.full((nx, ny), np.nan)

    for xi_str, xi_cells in cells.items():
        xi = int(xi_str)
        for yi_str, cell_data in xi_cells.items():
            yi = int(yi_str)
            if xi < nx and yi < ny:
                if "value" in cell_data:
                    grid[xi, yi] = cell_data["value"]
                elif "repetitions" in cell_data:
                    grid[xi, yi] = np.median(cell_data["repetitions"])

    return grid

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
Z_vqe = extract_grid(vqe_iqpe_data, "vqe")
Z_iqpe = extract_grid(vqe_iqpe_data, "iqpe")
Z_dmrg = extract_grid_from_result(dmrg_data, "dmrg")

print(f"VQE shape: {Z_vqe.shape}, NaN count: {np.isnan(Z_vqe).sum()}")
print(f"IQPE shape: {Z_iqpe.shape}, NaN count: {np.isnan(Z_iqpe).sum()}")
print(f"DMRG shape: {Z_dmrg.shape}, NaN count: {np.isnan(Z_dmrg).sum()}")

# Create VQE heatmap using qbp plotting function
plot_analytic_heatmap(
    x_values,
    y_values,
    x_label=r"$M$",
    y_label=r"$\phi$",
    Z=Z_vqe,
    output_path=f"{output_dir}/vqe-heatmap-E-M-vs-phi.pdf",
    hide_plot=False,
    z_label="Energy"
)

print(f"VQE heatmap saved to {output_dir}/vqe-heatmap-E-M-vs-phi.pdf")

# Create DMRG heatmap using qbp plotting function
plot_analytic_heatmap(
    x_values,
    y_values,
    x_label=r"$M$",
    y_label=r"$\phi$",
    Z=Z_dmrg,
    output_path=f"{output_dir}/dmrg-heatmap-E-M-vs-phi.pdf",
    hide_plot=False,
    z_label="Energy"
)

print(f"DMRG heatmap saved to {output_dir}/dmrg-heatmap-E-M-vs-phi.pdf")
