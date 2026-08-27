#!/usr/bin/env python3
"""
Add analytical reference lines to TFIM 1D h-sweep-Lx8 IQPE plots.
"""

import json
import numpy as np
import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qbp._plotting import _apply_rcparams, _save_and_show

# Directories
data_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/logs/tfim/1d/ground-state-energy/h-sweep-Lx8"
plot_dir = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/tfim/1d/ground-state-energy/h-sweep-Lx8"
analytic_data_path = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/tfim/pbc/tfim-1d-pbc-E-Lx-vs-h-analytic.json"

# Load analytical data
with open(analytic_data_path) as f:
    analytic_data = json.load(f)

# Extract Lx=8 analytical values (it's the 5th element, index 4)
analytic_result = analytic_data["result"]["analytic"]
analytic_x_vals = analytic_data["x_values"]  # Lx values
analytic_y_vals = analytic_data["y_values"]  # h values

# Find index for Lx=8
lx8_idx = 4  # Lx values: [4, 5, 6, 7, 8, 9, 10, 12]
if 8.0 in analytic_x_vals:
    lx8_idx = analytic_x_vals.index(8.0)

# Extract Lx=8 analytical values for all h
analytic_lx8_vals = {}
for h_idx, h_val in enumerate(analytic_y_vals):
    if str(lx8_idx) in analytic_result and str(h_idx) in analytic_result[str(lx8_idx)]:
        analytic_lx8_vals[h_val] = analytic_result[str(lx8_idx)][str(h_idx)]

print(f"Loaded analytical values for Lx=8:")
print(f"  H values: {list(analytic_lx8_vals.keys())}")
print(f"  Energy values: {list(analytic_lx8_vals.values())}")

# Get list of time parameters from log directory
import os
time_params = []
for f in os.listdir(data_dir):
    if f.startswith("tfim-1d-pbc-Lx8-E-h-iqpe-") and f.endswith(".json"):
        t_str = f.split("t")[-1].replace(".json", "")
        time_params.append(float(t_str))

time_params.sort()
print(f"\nFound {len(time_params)} time parameters: {time_params}")

_apply_rcparams()

# Process each time parameter
for t_param in time_params:
    json_file = f"{data_dir}/tfim-1d-pbc-Lx8-E-h-iqpe-t{t_param}.json"
    pdf_file = f"{plot_dir}/tfim-1d-pbc-Lx8-E-h-iqpe-t{t_param}.pdf"

    # Load IQPE data
    with open(json_file) as f:
        iqpe_data = json.load(f)

    h_values = iqpe_data["x_values"]
    iqpe_result = iqpe_data["result"]["iqpe"]

    # Extract IQPE values
    iqpe_vals = []
    for idx in range(len(h_values)):
        if str(idx) in iqpe_result:
            iqpe_vals.append(iqpe_result[str(idx)])
        else:
            iqpe_vals.append(np.nan)

    # Extract analytical values for the same h values
    analytic_vals = [analytic_lx8_vals.get(h, np.nan) for h in h_values]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot IQPE data
    ax.plot(h_values, iqpe_vals, marker='o', linestyle='-', linewidth=2, markersize=8,
            color='#0072B2', label='IQPE', zorder=3)

    # Plot analytical reference
    ax.plot(h_values, analytic_vals, marker='s', linestyle='--', linewidth=2, markersize=6,
            color='#888888', label='Analytic', zorder=2)

    ax.set_xlabel(r"$h$", fontsize=13, labelpad=8)
    ax.set_ylabel(r"$E$", fontsize=13, labelpad=8)
    ax.set_title(f"TFIM 1D (Lx=8, PBC) - Ground State Energy (t={t_param})", fontsize=13, pad=10)

    # Styling
    ax.grid(True, linestyle='--', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(direction="out", length=4, color="#888888")

    # Legend
    ax.legend(loc='best', fontsize=11, frameon=True, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    _save_and_show(fig, pdf_file, hide_plot=True)
    print(f"Saved: {pdf_file}")

print("\nDone!")
