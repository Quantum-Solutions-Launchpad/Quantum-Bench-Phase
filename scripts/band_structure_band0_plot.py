#!/usr/bin/env python3
"""Generate Band 0 heatmap for band structure analytic data."""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

import qbp

eval_dir = "/pscratch/sd/m/mbao202/NNL-P7/evaluation/band-structure"
output_path = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/new-data/band-structure/E-kx-vs-ky-band0-analytic.pdf"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

print("Generating Band 0 heatmap for Band Structure (Analytic)...")

try:
    # Load analytic band structure data
    log_path = f"{eval_dir}/simulated-noisy-analytic-E-kx-vs-ky.json"
    result = qbp.load_result(log_path)

    # Extract Band 0 data
    with open(log_path) as f:
        data = json.load(f)

    x_values = data["x_values"]
    y_values = data["y_values"]

    # Get Band 0 (ground state) from analytic results
    analytic_result = data["result"]["analytic"]

    # Extract Band 0 grid
    import numpy as np
    nx, ny = len(x_values), len(y_values)
    band0_grid = np.full((nx, ny), np.nan)

    for band_idx in range(1):  # Only Band 0
        for xi_str, xi_data in analytic_result.items():
            xi = int(xi_str)
            for yi_str, energies in xi_data.items():
                yi = int(yi_str)
                if xi < nx and yi < ny:
                    # energies is a list of energy values for each band
                    # Band 0 is the first (ground state)
                    if isinstance(energies, (list, tuple)) and len(energies) > 0:
                        band0_grid[xi, yi] = energies[0]

    print(f"  Band 0 grid shape: {band0_grid.shape}")
    print(f"  Min energy: {np.nanmin(band0_grid):.6f}")
    print(f"  Max energy: {np.nanmax(band0_grid):.6f}")

    # Create a new result with only Band 0 data
    result.raw["result"]["analytic"] = {}
    for xi in range(nx):
        result.raw["result"]["analytic"][str(xi)] = {}
        for yi in range(ny):
            result.raw["result"]["analytic"][str(xi)][str(yi)] = band0_grid[xi, yi]

    # Update grids
    result.grids["analytic"] = band0_grid

    # Plot using QBP's native plotting
    result.plot(output_path=output_path, hide_plot=True, hide_legend=False)

    print(f"✓ Saved: {output_path}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Band 0 heatmap generated!")
