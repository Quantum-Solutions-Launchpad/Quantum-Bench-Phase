#!/usr/bin/env python3
"""Regenerate ALL error plots in evaluation with large fonts."""

import sys
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')
sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7/scripts/slurm')

from pathlib import Path
import importlib.util

# Load error_plotting_utils from the phase-diagrams directory
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

# Define all evaluation directories and their configurations
configs = [
    # Magnetization M_total
    {
        "name": "Magnetization M_total",
        "eval_dir": f"{eval_base}/magnetization/M_total",
        "analytical_file": "simulated-ideal-analytic-M_total-n_occ-vs-U.json",
        "vqe_iqpe_file": "vqe-M_total-heatmap-n_occ-vs-U.json",
        "methods": {
            "vqe": {
                "json_key": "vqe",
                "label": "VQE",
                "color": "#1f77b4",
                "marker": "o",
                "size": 50,
            },
        },
        "x_label": r"$N_{occ}$",
        "y_label": r"$U$",
        "base_filename": "M_total_error",
    },
    # Magnetization M_stag
    {
        "name": "Magnetization M_stag",
        "eval_dir": f"{eval_base}/magnetization/M_stag",
        "analytical_file": "simulated-ideal-analytic-M_stag-n_occ-vs-U.json",
        "vqe_iqpe_file": "vqe-M_stag-heatmap-n_occ-vs-U.json",
        "methods": {
            "vqe": {
                "json_key": "vqe",
                "label": "VQE",
                "color": "#1f77b4",
                "marker": "o",
                "size": 50,
            },
        },
        "x_label": r"$N_{occ}$",
        "y_label": r"$U$",
        "base_filename": "M_stag_error",
    },
    # M-vs-phi
    {
        "name": "M-vs-phi",
        "eval_dir": f"{eval_base}/M-vs-phi",
        "analytical_file": "simulated-ideal-analytic-dmrg-E-M-vs-phi.json",
        "vqe_iqpe_file": "simulated-ideal-vqe-iqpe-E-M-vs-phi.json",
        "methods": {
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
        },
        "x_label": r"$\phi$",
        "y_label": r"$M$",
        "base_filename": "E_M_vs_phi",
    },
    # nocc-vs-t2
    {
        "name": "nocc-vs-t2",
        "eval_dir": f"{eval_base}/nocc-vs-t2",
        "analytical_file": "simulated-ideal-analytic-dmrg-E-nocc-vs-t2.json",
        "vqe_iqpe_file": "simulated-ideal-vqe-iqpe-E-nocc-vs-t2.json",
        "methods": {
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
        },
        "x_label": r"$N_{occ}$",
        "y_label": r"$t_2$",
        "base_filename": "E_nocc_vs_t2",
    },
    # nocc-vs-t2-hard-wall
    {
        "name": "nocc-vs-t2-hard-wall",
        "eval_dir": f"{eval_base}/nocc-vs-t2-hard-wall",
        "analytical_file": "simulated-ideal-analytic-dmrg-E-nocc-vs-t2-hard-wall.json",
        "vqe_iqpe_file": "simulated-ideal-vqe-iqpe-E-nocc-vs-t2-hard-wall.json",
        "methods": {
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
        },
        "x_label": r"$N_{occ}$",
        "y_label": r"$t_2$",
        "base_filename": "E_nocc_vs_t2_hard_wall",
    },
    # TFIM OBC
    {
        "name": "TFIM OBC",
        "eval_dir": f"{eval_base}/tfim/obc",
        "analytical_file": "simulated-noisy-analytic-dmrg-E-Lx-vs-h.json",
        "vqe_iqpe_file": "simulated-noisy-vqe-iqpe-E-Lx-vs-h.json",
        "methods": {
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
        },
        "x_label": r"$L_x$",
        "y_label": r"$h$",
        "base_filename": "E_tfim_obc",
    },
    # TFIM PBC
    {
        "name": "TFIM PBC",
        "eval_dir": f"{eval_base}/tfim/pbc",
        "analytical_file": "simulated-noisy-analytic-dmrg-E-Lx-vs-h.json",
        "vqe_iqpe_file": "simulated-noisy-vqe-iqpe-E-Lx-vs-h.json",
        "methods": {
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
        },
        "x_label": r"$L_x$",
        "y_label": r"$h$",
        "base_filename": "E_tfim_pbc",
    },
    # Band structure
    {
        "name": "Band Structure",
        "eval_dir": f"{eval_base}/band-structure",
        "analytical_file": "simulated-noisy-analytic-dmrg-E-kx-vs-ky.json",
        "vqe_iqpe_file": "simulated-noisy-vqe-iqpe-E-kx-vs-ky.json",
        "methods": {
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
        },
        "x_label": r"$k_x$",
        "y_label": r"$k_y$",
        "base_filename": "E_band-structure",
    },
]

# Regenerate all error plots
print("=" * 70)
print("Regenerating ALL error plots with large fonts")
print("=" * 70)

for config in configs:
    name = config["name"]
    eval_dir = config["eval_dir"]
    output_dir = f"{eval_dir}/errors"

    analytical_path = f"{eval_dir}/{config['analytical_file']}"
    vqe_iqpe_path = f"{eval_dir}/{config['vqe_iqpe_file']}"

    # Check if files exist
    if not Path(analytical_path).exists():
        print(f"\n✗ {name}: {config['analytical_file']} not found")
        continue
    if not Path(vqe_iqpe_path).exists():
        print(f"\n✗ {name}: {config['vqe_iqpe_file']} not found")
        continue

    print(f"\n{'─' * 70}")
    print(f"Regenerating: {name}")
    print(f"{'─' * 70}")

    try:
        # Load data
        x_values, y_values, analytical, method_data_dict = load_and_prepare_data(
            vqe_iqpe_path,
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

print("\n" + "=" * 70)
print("All error plots regenerated with large fonts!")
print("=" * 70)
