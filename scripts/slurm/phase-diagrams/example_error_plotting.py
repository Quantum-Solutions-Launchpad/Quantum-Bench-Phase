#!/usr/bin/env python3
"""
Example usage of the generalized error plotting utilities.

This script demonstrates how to compare DMRG, VQE, and IQPE methods
against analytical values with error analysis.
"""

import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

from error_plotting_utils import (
    load_and_prepare_data,
    plot_error_comparison,
    compare_all_methods,
)

# Configuration
base_dir = "/pscratch/sd/m/mbao202/NNL-P7"
evaluation_dir = base_dir + "/evaluation/nocc-vs-t2-hard-wall"
output_dir = evaluation_dir + "/errors"

# Data files from evaluation directory (combined analytic+dmrg and separate vqe+iqpe)
analytical_dmrg_path = evaluation_dir + "/simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json"
vqe_iqpe_path = evaluation_dir + "/simulated-ideal-vqe-iqpe-E-nocc-vs-t2-hard-wall.json"

# Define methods to compare
methods = {
    "dmrg": {
        "json_key": "dmrg",
        "label": "DMRG",
        "color": "#D7277C",
        "marker": "s",
        "size": 45,
    },
    "vqe": {
        "json_key": "vqe",
        "label": "VQE",
        "color": "#1f77b4",
        "marker": "o",
        "size": 50,
    },
    "iqpe": {
        "json_key": "iqpe",
        "label": "IQPE",
        "color": "#ff7f0e",
        "marker": "^",
        "size": 50,
    },
}

print("Loading data...")
x_values, y_values, analytical, method_data_dict = load_and_prepare_data(
    vqe_iqpe_path,
    analytical_dmrg_path,
    methods,
)

print(f"X values: {len(x_values)}, Y values: {len(y_values)}")
print(f"Analytical grid shape: {analytical.shape}")

print("\nGenerating error plots...")
errors = plot_error_comparison(
    x_values,
    y_values,
    analytical,
    method_data_dict,
    output_dir,
    x_label=r"$N_{occ}$",
    y_label=r"$t_2$",
    base_filename="E_nocc_vs_t2_hard_wall",
)

print("\nGenerating combined comparison plots...")
compare_all_methods(
    x_values,
    y_values,
    analytical,
    method_data_dict,
    output_dir,
    x_label=r"$N_{occ}$",
    y_label=r"$t_2$",
    base_filename="E_nocc_vs_t2_hard_wall",
)

print("\nDone!")
