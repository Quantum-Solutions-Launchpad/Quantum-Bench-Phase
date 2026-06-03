from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from quaph._dmrg import run_dmrg_itensor
from quaph._run import run_simulated_ideal


def _lattice_tag(lattice) -> str:
    return "x".join(str(x) for x in lattice)


def _file_tag(x_param: str, y_param: str | None) -> str:
    if y_param is None:
        return f"compare-{x_param}"
    return f"compare-{x_param}-vs-{y_param}"


def _grid_from_block(block, nx: int, ny: int, *, key: str | None = None) -> np.ndarray:
    values = np.full((nx, ny), np.nan)
    for ix in range(nx):
        row = block[str(ix)]
        for iy in range(ny):
            cell = row if ny == 1 and not isinstance(row, dict) else row[str(iy)]
            if key is not None:
                cell = cell[key]
            values[ix, iy] = float(cell)
    return values


def _block_from_grid(values: np.ndarray, *, is_1d: bool):
    if is_1d:
        return {ix: float(values[ix, 0]) for ix in range(values.shape[0])}
    return {
        ix: {iy: float(values[ix, iy]) for iy in range(values.shape[1])}
        for ix in range(values.shape[0])
    }


def _plot_subdir(plot_dir, model_name, lattice):
    return Path(plot_dir) / model_name / _lattice_tag(lattice)


def _log_subdir(log_dir, model_name, lattice):
    return Path(log_dir) / model_name / _lattice_tag(lattice)


def run_compare(
    model,
    *,
    lattice,
    x_param: str | None,
    x_range,
    y_param: str | None,
    y_range,
    n_occ: int | None,
    model_params: dict,
    algorithms: list[str],
    vqe_iters: int | None = None,
    vqe_layers: int | None = None,
    vqe_reps: int | None = None,
    iqpe_time: float | None = None,
    iqpe_trot: int | None = None,
    iqpe_iters: int | None = None,
    iqpe_reps: int | None = None,
    dmrg_julia: str = "julia",
    dmrg_julia_module: str | None = "julia/1.11.7",
    dmrg_julia_project: str | None = None,
    dmrg_script: str | None = None,
    dmrg_nsweeps: int = 4,
    dmrg_maxdims: str = "20,50,100,200",
    dmrg_cutoff: float = 1e-9,
    dmrg_seed: int = 1234,
    dmrg_conserve_qns: bool = True,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
    task_index: int | None = None,
    task_count: int = 1,
    prepare_only: bool = False,
    aggregate_only: bool = False,
    no_progress_log: bool = False,
):
    algorithms = [a.lower() for a in algorithms]
    allowed = {"exact", "vqe", "iqpe", "dmrg"}
    unknown = sorted(set(algorithms) - allowed)
    if unknown:
        raise ValueError(f"Unknown compare algorithms: {', '.join(unknown)}")
    if "exact" not in algorithms:
        algorithms = ["exact", *algorithms]
    if "dmrg" in algorithms and lattice is None:
        raise ValueError("DMRG compare runs require --lattice.")
    if "vqe" in algorithms and (vqe_iters is None or vqe_layers is None):
        raise ValueError("Compare runs with VQE require --vqe-iters and --vqe-layers.")
    if "iqpe" in algorithms and (iqpe_time is None or iqpe_trot is None or iqpe_iters is None):
        raise ValueError("Compare runs with IQPE require --iqpe-time, --iqpe-trot, and --iqpe-iters.")

    sim_result = run_simulated_ideal(
        model,
        lattice=lattice,
        x_param=x_param,
        x_range=x_range,
        y_param=y_param,
        y_range=y_range,
        n_occ=n_occ,
        model_params=model_params,
        vqe_iters=vqe_iters,
        vqe_layers=vqe_layers,
        vqe_reps=vqe_reps if "vqe" in algorithms else 0,
        iqpe_time=iqpe_time,
        iqpe_trot=iqpe_trot,
        iqpe_iters=iqpe_iters,
        iqpe_reps=iqpe_reps if "iqpe" in algorithms else 0,
        log_dir=log_dir,
        plot_dir=None,
        hide_plot=True,
        hide_legend=hide_legend,
        task_index=task_index,
        task_count=task_count,
        prepare_only=prepare_only,
        aggregate_only=aggregate_only,
        no_progress_log=no_progress_log,
    )

    dmrg_summary = None
    if "dmrg" in algorithms:
        dmrg_summary = run_dmrg_itensor(
            model,
            lattice=lattice,
            x_param=x_param,
            x_range=x_range,
            y_param=y_param,
            y_range=y_range,
            n_occ=n_occ,
            model_params=model_params,
            log_dir=log_dir,
            plot_dir=None,
            hide_plot=True,
            julia=dmrg_julia,
            julia_module=dmrg_julia_module,
            julia_project=dmrg_julia_project,
            nsweeps=dmrg_nsweeps,
            maxdims=dmrg_maxdims,
            cutoff=dmrg_cutoff,
            seed=dmrg_seed,
            conserve_qns=dmrg_conserve_qns,
            script_path=dmrg_script,
            task_index=task_index,
            task_count=task_count,
            prepare_only=prepare_only,
            aggregate_only=aggregate_only,
            no_progress_log=no_progress_log,
        )

    if task_index is not None:
        return {
            "type": "compare-shard",
            "task_index": task_index,
            "task_count": task_count,
            "algorithms": algorithms,
        }
    if prepare_only:
        return {
            "type": "compare-prepare",
            "algorithms": algorithms,
            "raw_log_path": sim_result.raw_log_path,
        }

    x_values = list(sim_result.x_values)
    y_values = list(sim_result.y_values)
    is_1d = not y_values
    nx = len(x_values)
    ny = 1 if is_1d else len(y_values)

    exact_values = np.asarray(sim_result.analytic_energies, dtype=float)
    if is_1d:
        exact_grid = exact_values.reshape(nx, 1)
    else:
        exact_grid = exact_values
    vqe_values = None if sim_result.vqe_best_energies is None else np.asarray(sim_result.vqe_best_energies, dtype=float)
    iqpe_values = None if sim_result.iqpe_best_energies is None else np.asarray(sim_result.iqpe_best_energies, dtype=float)

    dmrg_values = None
    if dmrg_summary is not None:
        dmrg_values = _grid_from_block(dmrg_summary["result"], nx, ny, key="energy")

    result_block = {"exact": _block_from_grid(exact_grid, is_1d=is_1d)}
    if vqe_values is not None:
        result_block["vqe"] = _block_from_grid(vqe_values.reshape(nx, ny), is_1d=is_1d)
    if iqpe_values is not None:
        result_block["iqpe"] = _block_from_grid(iqpe_values.reshape(nx, ny), is_1d=is_1d)
    if dmrg_values is not None:
        result_block["dmrg"] = _block_from_grid(dmrg_values, is_1d=is_1d)

    summary = {
        "type": "compare",
        "plot_format": "2d" if is_1d else "3d",
        "algorithms": algorithms,
        "model": model.name,
        "lattice": list(lattice) if lattice is not None else None,
        "x_param": x_param,
        "y_param": y_param,
        "x_values": x_values,
        "y_values": y_values,
        "parameters": {
            "model_params": model_params,
            "n_occ": n_occ,
        },
        "result": result_block,
        "source_logs": {
            "simulated": sim_result.summary_log_path,
            "dmrg": None if dmrg_summary is None else dmrg_summary.get("summary_path"),
        },
    }

    summary_path = None
    if log_dir is not None:
        log_subdir = _log_subdir(log_dir, model.name, lattice)
        os.makedirs(log_subdir, exist_ok=True)
        summary_path = log_subdir / f"{_file_tag(x_param, y_param)}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        summary["summary_path"] = str(summary_path)

    plot_path = None
    if plot_dir is not None:
        plot_subdir = _plot_subdir(plot_dir, model.name, lattice)
        os.makedirs(plot_subdir, exist_ok=True)
        plot_path = plot_subdir / f"{_file_tag(x_param, y_param)}.pdf"

    if plot_path is not None or not hide_plot:
        from quaph._plotting import plot_simulated

        extra_series = []
        if dmrg_values is not None:
            extra_series.append({
                "label": "DMRG",
                "values": dmrg_values[:, 0] if is_1d else dmrg_values,
                "color": "#D55E00",
                "marker": "D",
            })
        plot_simulated(
            np.asarray(x_values, dtype=float),
            [] if is_1d else np.asarray(y_values, dtype=float),
            model.param_labels.get(x_param, x_param),
            "$E$" if is_1d else model.param_labels.get(y_param, y_param),
            exact_grid[:, 0] if is_1d else exact_grid,
            None if vqe_values is None else vqe_values.reshape(nx, ny)[:, 0] if is_1d else vqe_values,
            None if iqpe_values is None else iqpe_values.reshape(nx, ny)[:, 0] if is_1d else iqpe_values,
            plot_format="2d" if is_1d else "3d",
            hide_legend=hide_legend,
            output_path=str(plot_path) if plot_path is not None else None,
            hide_plot=hide_plot,
            extra_series=extra_series,
        )
        if plot_path is not None:
            summary["plot_path"] = str(plot_path)
            if summary_path is not None:
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

    return summary
