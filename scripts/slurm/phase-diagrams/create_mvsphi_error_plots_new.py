#!/usr/bin/env python3
"""
Create error plots comparing VQE and DMRG methods against analytical values
for the M-vs-phi Haldane model.
"""

import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

from error_plotting_utils import (
    extract_grid,
    compute_errors,
    plot_error_comparison,
    compare_all_methods,
    MethodData,
)

# Configuration
base_dir = "/pscratch/sd/m/mbao202/NNL-P7"
evaluation_dir = base_dir + "/evaluation/M-vs-phi"
output_dir = evaluation_dir + "/errors"

# Data files - separate files for each method
analytic_path = evaluation_dir + "/simulated-ideal-analytic-E-M-vs-phi.json"
dmrg_path = evaluation_dir + "/simulated-ideal-dmrg-E-M-vs-phi.json"
vqe_path = evaluation_dir + "/simulated-ideal-vqe-E-M-vs-phi.json"

print("Loading data...")

# Load all three files
with open(analytic_path) as f:
    analytic_data = json.load(f)
with open(dmrg_path) as f:
    dmrg_data = json.load(f)
with open(vqe_path) as f:
    vqe_data = json.load(f)

# Get dimensions
x_values = analytic_data["x_values"]
y_values = analytic_data["y_values"]
nx, ny = len(x_values), len(y_values)

# Extract grids
analytical_grid = extract_grid(analytic_data, "analytic", nx, ny)
dmrg_grid = extract_grid(dmrg_data, "dmrg", nx, ny)
vqe_grid = extract_grid(vqe_data, "vqe", nx, ny)

print(f"X values: {len(x_values)}, Y values: {len(y_values)}")
print(f"Analytical grid shape: {analytical_grid.shape}")
print(f"DMRG grid shape: {dmrg_grid.shape}")
print(f"VQE grid shape: {vqe_grid.shape}")

# Create method data dictionary
method_data_dict = {
    "dmrg": MethodData(
        name="dmrg",
        values=dmrg_grid,
        label="DMRG",
        color="#D55E00",
        marker="D",
        size=45,
    ),
    "vqe": MethodData(
        name="vqe",
        values=vqe_grid,
        label="VQE",
        color="#0072B2",
        marker="o",
        size=50,
    ),
}

print("\nGenerating error plots...")
errors = plot_error_comparison(
    x_values,
    y_values,
    analytical_grid,
    method_data_dict,
    output_dir,
    x_label=r"$\phi$",
    y_label=r"$M$",
    base_filename="E_M_vs_phi",
)

print("\nGenerating combined comparison plots...")
compare_all_methods(
    x_values,
    y_values,
    analytical_grid,
    method_data_dict,
    output_dir,
    x_label=r"$\phi$",
    y_label=r"$M$",
    base_filename="E_M_vs_phi",
)

print("\nDone!")
