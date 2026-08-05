#!/usr/bin/env python3
import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')
# source /pscratch/sd/m/mbao202/NNL-P7/venv/bin/activate && python /pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/nocc_vs_t2_create_final_plot.py
from qbp._plotting import plot_simulated

# Data directory
data_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/logs/haldane/2x2/ground-state-energy/nocc-vs-t2"
output_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/haldane/2x2/ground-state-energy/nocc-vs-t2"

# Load analytical data
with open(f"{data_dir}/simulated-ideal-analytic-E-nocc-vs-t2.raw-data.json") as f:
    analytic_data = json.load(f)

# Load DMRG data (using .json for complete processed results)
with open(f"{data_dir}/simulated-ideal-dmrg-E-nocc-vs-t2.json") as f:
    dmrg_data = json.load(f)

# Load VQE+IQPE data
with open(f"{data_dir}/simulated-ideal-vqe-iqpe-E-nocc-vs-t2.raw-data.json") as f:
    vqe_iqpe_data = json.load(f)

x_values = analytic_data["x_values"]
y_values = analytic_data["y_values"]
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
                    # Take median of repetitions
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
Z_analytic = extract_grid(analytic_data, "analytic")
Z_dmrg = extract_grid_from_result(dmrg_data, "dmrg")
Z_vqe = extract_grid(vqe_iqpe_data, "vqe")
Z_iqpe = extract_grid(vqe_iqpe_data, "iqpe")

print(f"Analytical shape: {Z_analytic.shape}, NaN count: {np.isnan(Z_analytic).sum()}")
print(f"DMRG shape: {Z_dmrg.shape}, NaN count: {np.isnan(Z_dmrg).sum()}")
print(f"VQE shape: {Z_vqe.shape}, NaN count: {np.isnan(Z_vqe).sum()}")
print(f"IQPE shape: {Z_iqpe.shape}, NaN count: {np.isnan(Z_iqpe).sum()}")

# Create plot with all methods
plot_simulated(
    x_values,
    y_values,
    x_label=r"$N_{occ}$",
    y_label=r"$t_2$",
    Z_exact=Z_analytic,
    Z_vqe=Z_vqe,
    Z_iqpe=Z_iqpe,
    plot_format="3d",
    vqe_label="VQE",
    iqpe_label="IQPE",
    surface_label="Analytic",
    output_path=f"{output_dir}/simulated-ideal-all-E-nocc-vs-t2.pdf",
    hide_plot=False,
    extra_series=[{
        "values": Z_dmrg,
        "label": "DMRG",
        "color": "#D7277C",
        "marker": "s",
        "size": 45,
    }],
)

print(f"Plot saved to {output_dir}/simulated-ideal-all-E-nocc-vs-t2.pdf")
