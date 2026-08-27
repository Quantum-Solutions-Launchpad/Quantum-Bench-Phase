#!/usr/bin/env python3
"""Regenerate all PNG error plots in evaluation with large fonts."""

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

load_and_prepare_data = error_plotting_utils.load_and_prepare_data
plot_error_comparison = error_plotting_utils.plot_error_comparison
compare_all_methods = error_plotting_utils.compare_all_methods

base_dir = "/pscratch/sd/m/mbao202/NNL-P7"
eval_base = f"{base_dir}/evaluation"

# Define error plot configurations
configs = [
    # M-vs-phi
    {
        "name": "M-vs-phi",
        "eval_dir": f"{eval_base}/M-vs-phi",
        "analytical_file": "simulated-ideal-analytic-E-M-vs-phi.json",
        "dmrg_file": "simulated-ideal-dmrg-E-M-vs-phi.json",
        "vqe_file": "simulated-ideal-vqe-E-M-vs-phi.json",
        "methods": {
            "dmrg": {"json_key": "dmrg", "label": "DMRG", "color": "#D7277C", "marker": "s", "size": 45},
            "vqe": {"json_key": "vqe", "label": "VQE", "color": "#1f77b4", "marker": "o", "size": 50},
        },
        "x_label": r"$\phi$",
        "y_label": r"$M$",
        "base_filename": "E_M_vs_phi",
        "separate_files": True,
    },
    # nocc-vs-t2
    {
        "name": "nocc-vs-t2",
        "eval_dir": f"{eval_base}/nocc-vs-t2",
        "analytical_file": "simulated-ideal-analytic-E-nocc-vs-t2.json",
        "dmrg_file": "simulated-ideal-dmrg-E-nocc-vs-t2.json",
        "vqe_file": "simulated-ideal-vqe-iqpe-E-nocc-vs-t2.json",
        "methods": {
            "dmrg": {"json_key": "dmrg", "label": "DMRG", "color": "#D7277C", "marker": "s", "size": 45},
            "vqe": {"json_key": "vqe", "label": "VQE", "color": "#1f77b4", "marker": "o", "size": 50},
            "iqpe": {"json_key": "iqpe", "label": "IQPE", "color": "#ff7f0e", "marker": "^", "size": 50},
        },
        "x_label": r"$N_{occ}$",
        "y_label": r"$t_2$",
        "base_filename": "E_nocc_vs_t2",
        "separate_files": True,
    },
    # TFIM OBC
    {
        "name": "TFIM OBC",
        "eval_dir": f"{eval_base}/tfim/obc",
        "analytical_file": "simulated-noisy-analytic-E-Lx-vs-h.json",
        "dmrg_file": "simulated-noisy-dmrg-E-Lx-vs-h.json",
        "vqe_file": "simulated-noisy-vqe-iqpe-E-Lx-vs-h.json",
        "methods": {
            "dmrg": {"json_key": "dmrg", "label": "DMRG", "color": "#D7277C", "marker": "s", "size": 45},
            "vqe": {"json_key": "vqe", "label": "VQE", "color": "#1f77b4", "marker": "o", "size": 50},
            "iqpe": {"json_key": "iqpe", "label": "IQPE", "color": "#ff7f0e", "marker": "^", "size": 50},
        },
        "x_label": r"$L_x$",
        "y_label": r"$h$",
        "base_filename": "E_tfim_obc",
        "separate_files": True,
    },
    # TFIM PBC
    {
        "name": "TFIM PBC",
        "eval_dir": f"{eval_base}/tfim/pbc",
        "analytical_file": "simulated-noisy-analytic-E-Lx-vs-h.json",
        "dmrg_file": "simulated-noisy-dmrg-E-Lx-vs-h.json",
        "vqe_file": "simulated-noisy-vqe-iqpe-E-Lx-vs-h.json",
        "methods": {
            "dmrg": {"json_key": "dmrg", "label": "DMRG", "color": "#D7277C", "marker": "s", "size": 45},
            "vqe": {"json_key": "vqe", "label": "VQE", "color": "#1f77b4", "marker": "o", "size": 50},
            "iqpe": {"json_key": "iqpe", "label": "IQPE", "color": "#ff7f0e", "marker": "^", "size": 50},
        },
        "x_label": r"$L_x$",
        "y_label": r"$h$",
        "base_filename": "E_tfim_pbc",
        "separate_files": True,
    },
    # Band Structure
    {
        "name": "Band Structure",
        "eval_dir": f"{eval_base}/band-structure",
        "analytical_file": "simulated-noisy-analytic-E-kx-vs-ky.json",
        "dmrg_file": "simulated-noisy-dmrg-E-kx-vs-ky.json",
        "vqe_file": "simulated-noisy-vqe-iqpe-E-kx-vs-ky.json",
        "methods": {
            "dmrg": {"json_key": "dmrg", "label": "DMRG", "color": "#D7277C", "marker": "s", "size": 45},
            "vqe": {"json_key": "vqe", "label": "VQE", "color": "#1f77b4", "marker": "o", "size": 50},
            "iqpe": {"json_key": "iqpe", "label": "IQPE", "color": "#ff7f0e", "marker": "^", "size": 50},
        },
        "x_label": r"$k_x$",
        "y_label": r"$k_y$",
        "base_filename": "E_band-structure",
        "separate_files": True,
    },
]

print("=" * 70)
print("Regenerating PNG error plots with large fonts")
print("=" * 70)

for config in configs:
    name = config["name"]
    eval_dir = config["eval_dir"]
    output_dir = f"{eval_dir}/errors"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analytical_path = f"{eval_dir}/{config['analytical_file']}"

    if not Path(analytical_path).exists():
        print(f"\n✗ {name}: {config['analytical_file']} not found")
        continue

    print(f"\n{'─' * 70}")
    print(f"Regenerating: {name}")
    print(f"{'─' * 70}")

    try:
        # For separate files, merge analytic and dmrg first
        if config.get("separate_files"):
            dmrg_path = f"{eval_dir}/{config['dmrg_file']}"
            vqe_path = f"{eval_dir}/{config['vqe_file']}"

            # Check if files exist
            if not Path(dmrg_path).exists():
                print(f"  (Note: DMRG file not found, using VQE only)")
                config["methods"] = {k: v for k, v in config["methods"].items() if k != "dmrg"}
                analytical_path_for_load = vqe_path
            else:
                analytical_path_for_load = analytical_path

            # Load data - merge analytic+dmrg
            import json
            with open(analytical_path) as f:
                analytic_data = json.load(f)
            with open(dmrg_path) as f:
                dmrg_data = json.load(f)

            # Merge dmrg into analytic
            if "result" not in analytic_data:
                analytic_data["result"] = {}
            analytic_data["result"]["dmrg"] = dmrg_data["result"]["dmrg"]

            # Now load with merged data
            x_values, y_values, analytical, method_data_dict = load_and_prepare_data(
                vqe_path,
                analytical_path,
                config["methods"],
                separate_analytic_dmrg=True,
                dmrg_path=dmrg_path,
            )
        else:
            x_values, y_values, analytical, method_data_dict = load_and_prepare_data(
                config["vqe_file"],
                analytical_path,
                config["methods"],
            )

        # Generate error plots
        errors = plot_error_comparison(
            x_values,
            y_values,
            analytical,
            method_data_dict,
            output_dir,
            x_label=config["x_label"],
            y_label=config["y_label"],
            base_filename=config["base_filename"],
        )

        # Generate comparison plots
        compare_all_methods(
            x_values,
            y_values,
            analytical,
            method_data_dict,
            output_dir,
            x_label=config["x_label"],
            y_label=config["y_label"],
            base_filename=config["base_filename"],
        )

        print(f"✓ {name} complete")

    except Exception as e:
        print(f"✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("All PNG error plots regenerated with large fonts!")
print("=" * 70)
