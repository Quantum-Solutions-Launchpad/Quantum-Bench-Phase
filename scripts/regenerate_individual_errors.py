#!/usr/bin/env python3
"""Regenerate INDIVIDUAL error plots (VQE, DMRG, IQPE separately) with large fonts."""

import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

# Load error_plotting_utils
spec = importlib.util.spec_from_file_location(
    "error_plotting_utils",
    "/pscratch/sd/m/mbao202/NNL-P7/scripts/slurm/phase-diagrams/error_plotting_utils.py"
)
error_plotting_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(error_plotting_utils)

plot_error_comparison = error_plotting_utils.plot_error_comparison
load_and_prepare_data = error_plotting_utils.load_and_prepare_data
extract_grid = error_plotting_utils.extract_grid
MethodData = error_plotting_utils.MethodData

base_dir = "/pscratch/sd/m/mbao202/NNL-P7"
eval_base = f"{base_dir}/evaluation"

configs = [
    {
        "name": "M-vs-phi",
        "eval_dir": f"{eval_base}/M-vs-phi",
        "analytical": "simulated-ideal-analytic-E-M-vs-phi.json",
        "dmrg": "simulated-ideal-dmrg-E-M-vs-phi.json",
        "vqe": "simulated-ideal-vqe-E-M-vs-phi.json",
        "x_label": r"$\phi$",
        "y_label": r"$M$",
        "base_filename": "E_M_vs_phi",
    },
    {
        "name": "nocc-vs-t2",
        "eval_dir": f"{eval_base}/nocc-vs-t2",
        "analytical": "simulated-ideal-analytic-E-nocc-vs-t2.json",
        "dmrg": "simulated-ideal-dmrg-E-nocc-vs-t2.json",
        "vqe": "simulated-ideal-vqe-iqpe-E-nocc-vs-t2.json",
        "x_label": r"$N_{occ}$",
        "y_label": r"$t_2$",
        "base_filename": "E_nocc_vs_t2",
    },
    {
        "name": "nocc-vs-t2-hard-wall",
        "eval_dir": f"{eval_base}/nocc-vs-t2-hard-wall",
        "analytical": "simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json",
        "vqe": "simulated-ideal-vqe-iqpe-E-nocc-vs-t2-hard-wall.json",
        "x_label": r"$N_{occ}$",
        "y_label": r"$t_2$",
        "base_filename": "E_nocc_vs_t2_hard_wall",
    },
    {
        "name": "TFIM OBC",
        "eval_dir": f"{eval_base}/tfim/obc",
        "analytical": "simulated-noisy-analytic-E-Lx-vs-h.json",
        "dmrg": "simulated-noisy-dmrg-E-Lx-vs-h.json",
        "vqe": "simulated-noisy-vqe-iqpe-E-Lx-vs-h.json",
        "x_label": r"$L_x$",
        "y_label": r"$h$",
        "base_filename": "E_tfim_obc",
    },
    {
        "name": "TFIM PBC",
        "eval_dir": f"{eval_base}/tfim/pbc",
        "analytical": "simulated-noisy-analytic-E-Lx-vs-h.json",
        "dmrg": "simulated-noisy-dmrg-E-Lx-vs-h.json",
        "vqe": "simulated-noisy-vqe-iqpe-E-Lx-vs-h.json",
        "x_label": r"$L_x$",
        "y_label": r"$h$",
        "base_filename": "E_tfim_pbc",
    },
    {
        "name": "Band Structure",
        "eval_dir": f"{eval_base}/band-structure",
        "analytical": "simulated-noisy-analytic-E-kx-vs-ky.json",
        "dmrg": "simulated-noisy-dmrg-E-kx-vs-ky.json",
        "vqe": "simulated-noisy-vqe-E-kx-vs-ky.json",
        "iqpe": "simulated-noisy-iqpe-E-kx-vs-ky-filtered.json",
        "x_label": r"$k_x$",
        "y_label": r"$k_y$",
        "base_filename": "E_band-structure",
    },
]

print("=" * 70)
print("Regenerating INDIVIDUAL error plots with large fonts")
print("=" * 70)

for config in configs:
    name = config["name"]
    eval_dir = config["eval_dir"]
    output_dir = f"{eval_dir}/errors"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analytical_path = f"{eval_dir}/{config['analytical']}"

    if not Path(analytical_path).exists():
        print(f"\n✗ {name}: {config['analytical']} not found")
        continue

    print(f"\n{'─' * 70}")
    print(f"Regenerating: {name}")
    print(f"{'─' * 70}")

    try:
        import json

        # Load analytical data
        with open(analytical_path) as f:
            analytical_data = json.load(f)

        x_values = analytical_data.get("x_values", [])
        y_values = analytical_data.get("y_values", [])
        nx, ny = len(x_values), len(y_values)

        # Extract analytical grid
        analytical_grid = extract_grid(analytical_data, "analytic", nx, ny)

        # Build methods dict based on available files
        methods = {}
        method_data_dict = {}

        # Check and load DMRG
        if "dmrg" in config:
            dmrg_path = f"{eval_dir}/{config['dmrg']}"
            if Path(dmrg_path).exists():
                with open(dmrg_path) as f:
                    dmrg_data = json.load(f)
                dmrg_grid = extract_grid(dmrg_data, "dmrg", nx, ny)
                methods["dmrg"] = {
                    "json_key": "dmrg",
                    "label": "DMRG",
                    "color": "#D7277C",
                    "marker": "s",
                    "size": 45,
                }
                method_data_dict["dmrg"] = MethodData(
                    name="dmrg",
                    values=dmrg_grid,
                    label="DMRG",
                    color="#D7277C",
                    marker="s",
                    size=45,
                )

        # Load VQE
        vqe_path = f"{eval_dir}/{config['vqe']}"
        if Path(vqe_path).exists():
            with open(vqe_path) as f:
                vqe_data = json.load(f)
            vqe_grid = extract_grid(vqe_data, "vqe", nx, ny)
            methods["vqe"] = {
                "json_key": "vqe",
                "label": "VQE",
                "color": "#1f77b4",
                "marker": "o",
                "size": 50,
            }
            method_data_dict["vqe"] = MethodData(
                name="vqe",
                values=vqe_grid,
                label="VQE",
                color="#1f77b4",
                marker="o",
                size=50,
            )

        # Load IQPE if available
        if "iqpe" in config:
            iqpe_path = f"{eval_dir}/{config['iqpe']}"
            if Path(iqpe_path).exists():
                with open(iqpe_path) as f:
                    iqpe_data = json.load(f)
                iqpe_grid = extract_grid(iqpe_data, "iqpe", nx, ny)
                methods["iqpe"] = {
                    "json_key": "iqpe",
                    "label": "IQPE",
                    "color": "#ff7f0e",
                    "marker": "^",
                    "size": 50,
                }
                method_data_dict["iqpe"] = MethodData(
                    name="iqpe",
                    values=iqpe_grid,
                    label="IQPE",
                    color="#ff7f0e",
                    marker="^",
                    size=50,
                )

        if not method_data_dict:
            print(f"✗ {name}: No method data found")
            continue

        # Generate individual error plots
        errors = plot_error_comparison(
            x_values,
            y_values,
            analytical_grid,
            method_data_dict,
            output_dir,
            x_label=config["x_label"],
            y_label=config["y_label"],
            base_filename=config["base_filename"],
        )

        print(f"✓ {name} individual plots complete")

    except Exception as e:
        print(f"✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("All individual error plots regenerated with large fonts!")
print("=" * 70)
