#!/usr/bin/env python3
"""Generate E plots for TFIM OBC and PBC using QBP's native plotting."""

import sys
import json
from pathlib import Path

sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

import qbp

eval_base = "/pscratch/sd/m/mbao202/NNL-P7/evaluation"
output_base = "/pscratch/sd/m/mbao202/NNL-P7/manuscript-plots/plots/new-data/tfim"
Path(output_base).mkdir(parents=True, exist_ok=True)

for variant in ["obc", "pbc"]:
    eval_dir = f"{eval_base}/tfim/{variant}"

    print(f"Generating E plot for TFIM {variant.upper()} using QBP...")

    try:
        # Load all three data sources
        analytic_path = f"{eval_dir}/tfim-1d-{variant}-E-Lx-vs-h-analytic.json"
        dmrg_path = f"{eval_dir}/tfim-1d-{variant}-E-Lx-vs-h-dmrg.json"
        vqe_path = f"{eval_dir}/tfim-1d-{variant}-E-Lx-vs-h-vqe.json"

        # Load results
        analytic_result = qbp.load_result(analytic_path)
        dmrg_result = qbp.load_result(dmrg_path)
        vqe_result = qbp.load_result(vqe_path)

        # Merge results into analytic's raw data
        analytic_result.raw["methods"] = ["analytic", "dmrg", "vqe"]
        analytic_result.raw["result"]["dmrg"] = dmrg_result.raw["result"]["dmrg"]
        analytic_result.raw["result"]["vqe"] = vqe_result.raw["result"]["vqe"]

        # Update methods and grids
        analytic_result.methods = ["analytic", "dmrg", "vqe"]
        analytic_result.grids["dmrg"] = dmrg_result.grids["dmrg"]
        analytic_result.grids["vqe"] = vqe_result.grids["vqe"]

        # Plot using QBP's native plotting with all three methods
        output_path = f"{output_base}/E-Lx-vs-h-{variant}.pdf"
        analytic_result.plot(output_path=output_path, hide_plot=True, hide_legend=False)

        print(f"✓ Saved: {output_path}")

    except Exception as e:
        print(f"✗ Error for TFIM {variant.upper()}: {e}")
        import traceback
        traceback.print_exc()

print("\n✓ All TFIM E plots generated with QBP format!")
