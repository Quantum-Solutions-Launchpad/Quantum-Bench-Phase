#!/usr/bin/env python3
"""Generalized error plotting utilities for comparing methods against analytical values."""

import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, '/pscratch/sd/m/mbao202/NNL-P7')

# Apply qbp-style rcParams
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})


@dataclass
class MethodData:
    """Container for method data and metadata."""
    name: str
    values: np.ndarray
    label: str
    color: str
    marker: str = "o"
    size: int = 50


def extract_grid(data: Dict, method_name: str, nx: int, ny: int, band_index: int = 0) -> np.ndarray:
    """
    Extract 2D grid from cells or result data.

    Args:
        data: Dictionary containing cells or result data
        method_name: Name of method to extract (e.g., "analytic", "dmrg", "vqe", "iqpe")
        nx, ny: Grid dimensions
        band_index: For band structure data with multiple values per point, which band to extract (default: 0 = lowest/first)

    Returns:
        2D numpy array with method values
    """
    # Handle both 'cells' (old format) and 'result' (new format)
    if "cells" in data:
        data_dict = data["cells"][method_name]
    elif "result" in data:
        data_dict = data["result"][method_name]
    else:
        raise ValueError("Data must contain either 'cells' or 'result' key")

    grid = np.full((nx, ny), np.nan)

    for xi_str, xi_data in data_dict.items():
        xi = int(xi_str)
        for yi_str, cell_value in xi_data.items():
            yi = int(yi_str)
            if xi < nx and yi < ny:
                # Handle different value formats
                if isinstance(cell_value, dict):
                    if "value" in cell_value:
                        grid[xi, yi] = cell_value["value"]
                    elif "energy" in cell_value:
                        grid[xi, yi] = cell_value["energy"]
                    elif "repetitions" in cell_value:
                        # Take median of repetitions
                        grid[xi, yi] = np.median(cell_value["repetitions"])
                elif isinstance(cell_value, list):
                    # For band structure data: extract specified band or first value
                    if band_index < len(cell_value):
                        grid[xi, yi] = cell_value[band_index]
                    else:
                        grid[xi, yi] = cell_value[0] if cell_value else np.nan
                else:
                    # Direct scalar value
                    grid[xi, yi] = cell_value

    return grid


def compute_errors(
    analytical: np.ndarray,
    method_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute absolute and relative errors.

    Args:
        analytical: Reference analytical values
        method_values: Computed method values

    Returns:
        Tuple of (absolute_error, relative_error)
    """
    absolute_error = np.abs(method_values - analytical)

    # For relative error: use NaN where analytical is too close to zero
    # This avoids artificially inflated errors from division by near-zero values
    relative_error = np.full_like(absolute_error, np.nan)

    # Only compute relative error where analytical is significantly non-zero
    valid_analytical = np.abs(analytical) > 1e-8
    relative_error[valid_analytical] = absolute_error[valid_analytical] / np.abs(analytical[valid_analytical])

    return absolute_error, relative_error


def merge_separate_data_files(
    analytic_path: str,
    dmrg_path: str,
) -> Dict:
    """
    Merge separate analytic and DMRG JSON files into a single data structure.

    Args:
        analytic_path: Path to analytic JSON file
        dmrg_path: Path to DMRG JSON file

    Returns:
        Merged dictionary with both analytic and dmrg in result
    """
    with open(analytic_path) as f:
        analytic_data = json.load(f)
    with open(dmrg_path) as f:
        dmrg_data = json.load(f)

    # Merge dmrg results into analytic data
    if "result" not in analytic_data:
        analytic_data["result"] = {}
    analytic_data["result"]["dmrg"] = dmrg_data["result"]["dmrg"]

    return analytic_data


def load_and_prepare_data(
    data_path: str,
    analytical_data_path: str,
    methods: Dict[str, Dict[str, str]],
    separate_analytic_dmrg: bool = False,
    dmrg_path: Optional[str] = None,
) -> Tuple[List[float], List[float], np.ndarray, Dict[str, MethodData]]:
    """
    Load data from files and prepare grids for error analysis.

    Args:
        data_path: Path to JSON file containing method data (VQE, IQPE, etc.)
        analytical_data_path: Path to JSON file containing analytical reference (or analytic if separate_analytic_dmrg=True)
        methods: Dictionary mapping method_key -> {
            'json_key': str (key in JSON cells/result),
            'label': str,
            'color': str,
            'marker': str (optional),
            'size': int (optional)
        }
        separate_analytic_dmrg: If True, analytical_data_path points to analytic file and dmrg_path to DMRG file
        dmrg_path: Path to DMRG JSON file (required if separate_analytic_dmrg=True)

    Returns:
        Tuple of (x_values, y_values, analytical_grid, method_data_dict)
    """
    # Load analytical data (with optional separate DMRG file)
    if separate_analytic_dmrg and dmrg_path:
        analytical_data = merge_separate_data_files(analytical_data_path, dmrg_path)
    else:
        with open(analytical_data_path) as f:
            analytical_data = json.load(f)

    # Load method data
    with open(data_path) as f:
        method_data = json.load(f)

    x_values = analytical_data["x_values"]
    y_values = analytical_data["y_values"]
    nx, ny = len(x_values), len(y_values)

    # Extract analytical grid
    analytical_grid = extract_grid(analytical_data, "analytic", nx, ny)

    # Determine which keys are available in each file
    analytical_keys = set(analytical_data.get("cells", {}).keys()) | set(analytical_data.get("result", {}).keys())
    method_keys = set(method_data.get("cells", {}).keys()) | set(method_data.get("result", {}).keys())

    # Extract method grids and compute errors
    method_data_dict = {}

    for method_key, config in methods.items():
        json_key = config["json_key"]
        label = config["label"]
        color = config["color"]
        marker = config.get("marker", "o")
        size = config.get("size", 50)

        # Determine which file has this method
        if json_key in analytical_keys:
            source_data = analytical_data
        elif json_key in method_keys:
            source_data = method_data
        else:
            raise ValueError(f"Method '{json_key}' not found in either data file")

        values = extract_grid(source_data, json_key, nx, ny)
        method_data_dict[method_key] = MethodData(
            name=method_key,
            values=values,
            label=label,
            color=color,
            marker=marker,
            size=size,
        )

    return x_values, y_values, analytical_grid, method_data_dict


def plot_heatmap(
    grid: np.ndarray,
    x_values: List[float],
    y_values: List[float],
    title: str,
    x_label: str,
    y_label: str,
    z_label: str,
    output_path: str,
    cmap: str = "viridis",
) -> None:
    """
    Create a heatmap plot of a 2D grid.

    Args:
        grid: 2D numpy array to plot
        x_values: X-axis values
        y_values: Y-axis values
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        z_label: Z-axis (colorbar) label
        output_path: Path to save the figure
        cmap: Colormap name
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(grid.T, origin="lower", cmap=cmap, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels([f"{x:.1f}" if isinstance(x, float) else str(x) for x in x_values], rotation=45, fontsize=11)
    ax.set_yticklabels([f"{y:.2f}" if isinstance(y, float) else str(y) for y in y_values], fontsize=11)

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=13)

    # Add colorbar with z-axis label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(z_label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_comparison(
    x_values: List[float],
    y_values: List[float],
    analytical: np.ndarray,
    method_data_dict: Dict[str, MethodData],
    output_dir: str,
    x_label: str = "X",
    y_label: str = "Y",
    base_filename: str = "error_comparison",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create error plots comparing methods against analytical reference.

    Args:
        x_values: X-axis values
        y_values: Y-axis values
        analytical: Analytical reference grid
        method_data_dict: Dictionary of MethodData for each method
        output_dir: Output directory for plots
        x_label: X-axis label
        y_label: Y-axis label
        base_filename: Base filename for output plots

    Returns:
        Dictionary mapping method_name -> (absolute_error, relative_error)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    errors = {}

    # Print summary statistics
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)

    for method_name, method_data in method_data_dict.items():
        abs_err, rel_err = compute_errors(analytical, method_data.values)
        errors[method_name] = (abs_err, rel_err)

        # Statistics (ignoring NaNs)
        valid_abs = abs_err[~np.isnan(abs_err)]
        valid_rel = rel_err[~np.isnan(rel_err)]

        print(f"\n{method_name}:")
        print(f"  Data shape: {method_data.values.shape}, NaN count: {np.isnan(method_data.values).sum()}")
        print(f"  Absolute Error - Mean: {np.mean(valid_abs):.6e}, Std: {np.std(valid_abs):.6e}, Max: {np.max(valid_abs):.6e}")
        print(f"  Relative Error - Mean: {np.mean(valid_rel):.6e}, Std: {np.std(valid_rel):.6e}, Max: {np.max(valid_rel):.6e}")

    print("\n" + "="*60)

    # Plot absolute errors for each method
    for method_name, method_data in method_data_dict.items():
        abs_err, _ = errors[method_name]

        output_path = f"{output_dir}/{base_filename}_absolute_{method_name}.png"
        plot_heatmap(
            abs_err,
            x_values,
            y_values,
            title=f"Absolute Error: {method_data.label}",
            x_label=x_label,
            y_label=y_label,
            z_label="Absolute Error",
            output_path=output_path,
            cmap="YlOrRd",
        )
        print(f"Absolute error plot saved: {output_path}")

    # Plot relative errors for each method (as percentage)
    for method_name, method_data in method_data_dict.items():
        _, rel_err = errors[method_name]
        rel_err_percent = rel_err * 100

        output_path = f"{output_dir}/{base_filename}_relative_{method_name}.png"
        plot_heatmap(
            rel_err_percent,
            x_values,
            y_values,
            title=f"Relative Error: {method_data.label}",
            x_label=x_label,
            y_label=y_label,
            z_label="Relative Error (%)",
            output_path=output_path,
            cmap="YlOrRd",
        )
        print(f"Relative error plot saved: {output_path}")

    return errors


def plot_comparison_overlay(
    x_values: List[float],
    y_values: List[float],
    errors: Dict[str, Tuple[np.ndarray, np.ndarray]],
    method_data_dict: Dict[str, MethodData],
    output_dir: str,
    x_label: str = "X",
    y_label: str = "Y",
    base_filename: str = "error_comparison",
) -> None:
    """
    Create comparison plots with all methods on one plot.

    Args:
        x_values: X-axis values
        y_values: Y-axis values
        errors: Dictionary mapping method_name -> (absolute_error, relative_error)
        method_data_dict: Dictionary of MethodData for each method
        output_dir: Output directory for plots
        x_label: X-axis label
        y_label: Y-axis label
        base_filename: Base filename for output plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create combined absolute error plot
    fig, axes = plt.subplots(1, len(errors), figsize=(5 * len(errors), 4))
    if len(errors) == 1:
        axes = [axes]

    for idx, (method_name, (abs_err, _)) in enumerate(errors.items()):
        method_data = method_data_dict[method_name]

        im = axes[idx].imshow(abs_err.T, origin="lower", cmap="YlOrRd", aspect="auto")

        axes[idx].set_xticks(np.arange(len(x_values)))
        axes[idx].set_yticks(np.arange(len(y_values)))
        axes[idx].set_xticklabels([f"{x:.1f}" if isinstance(x, float) else str(x) for x in x_values], rotation=45, fontsize=11)
        axes[idx].set_yticklabels([f"{y:.2f}" if isinstance(y, float) else str(y) for y in y_values], fontsize=11)

        axes[idx].set_xlabel(x_label, fontsize=13)
        axes[idx].set_ylabel(y_label, fontsize=13)
        axes[idx].set_title(f"{method_data.label}\nAbsolute Error", fontsize=13)

        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label("Absolute Error", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path = f"{output_dir}/{base_filename}_absolute_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined absolute error plot saved: {output_path}")

    # Create combined relative error plot (as percentage)
    fig, axes = plt.subplots(1, len(errors), figsize=(5 * len(errors), 4))
    if len(errors) == 1:
        axes = [axes]

    for idx, (method_name, (_, rel_err)) in enumerate(errors.items()):
        method_data = method_data_dict[method_name]
        rel_err_percent = rel_err * 100

        im = axes[idx].imshow(rel_err_percent.T, origin="lower", cmap="YlOrRd", aspect="auto")

        axes[idx].set_xticks(np.arange(len(x_values)))
        axes[idx].set_yticks(np.arange(len(y_values)))
        axes[idx].set_xticklabels([f"{x:.1f}" if isinstance(x, float) else str(x) for x in x_values], rotation=45, fontsize=11)
        axes[idx].set_yticklabels([f"{y:.2f}" if isinstance(y, float) else str(y) for y in y_values], fontsize=11)

        axes[idx].set_xlabel(x_label, fontsize=13)
        axes[idx].set_ylabel(y_label, fontsize=13)
        axes[idx].set_title(f"{method_data.label}\nRelative Error (%)", fontsize=13)

        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label("Relative Error (%)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path = f"{output_dir}/{base_filename}_relative_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined relative error plot saved: {output_path}")


def compare_all_methods(
    x_values: List[float],
    y_values: List[float],
    analytical: np.ndarray,
    method_data_dict: Dict[str, MethodData],
    output_dir: str,
    x_label: str = "X",
    y_label: str = "Y",
    base_filename: str = "error_comparison",
) -> None:
    """
    Create comparison plots showing all methods on one plot.

    Args:
        x_values: X-axis values
        y_values: Y-axis values
        analytical: Analytical reference grid
        method_data_dict: Dictionary of MethodData for each method
        output_dir: Output directory for plots
        x_label: X-axis label
        y_label: Y-axis label
        base_filename: Base filename for output plots
    """
    errors = {}

    for method_name, method_data in method_data_dict.items():
        abs_err, rel_err = compute_errors(analytical, method_data.values)
        errors[method_name] = (abs_err, rel_err)

    plot_comparison_overlay(
        x_values,
        y_values,
        errors,
        method_data_dict,
        output_dir,
        x_label=x_label,
        y_label=y_label,
        base_filename=base_filename,
    )
