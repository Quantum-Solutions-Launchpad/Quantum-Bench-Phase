from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from qiskit_nature.second_q.mappers import JordanWignerMapper

from quaph._model import Model
from quaph._core import (
    resolve_sweep,
    analytic, vqe, iqpe,
    vqe_other_benchmarks, iqpe_other_benchmarks,
)
from quaph._plotting import plot_analytic, plot_simulated
from quaph._registry import get_model as _get_model


def _resolve_model(model) -> Model:
    if isinstance(model, str):
        return _get_model(model)
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance or model name string, got {type(model).__name__}")
    return model


def _resolve_sweep_params(model: Model, x_param, x_range, y_param, y_range):
    sd = model.sweep_defaults
    _N_OCC_DEFAULT = {"param": "n_occ", "range": None}
    x_def = sd.get("x", _N_OCC_DEFAULT)
    y_def = sd.get("y", _N_OCC_DEFAULT)
    if x_param is None:
        x_param = x_def["param"]
    if x_range is None:
        x_range = x_def.get("range")
    if y_param is None:
        y_param = y_def["param"]
    if y_range is None:
        y_range = y_def.get("range")
    return x_param, x_range, y_param, y_range


def _log_subdir(log_dir, model_name, n_sites):
    return os.path.join(log_dir, model_name, f"{n_sites}-sites")


def _plot_subdir(plot_dir, model_name, n_sites):
    return os.path.join(plot_dir, model_name, f"{n_sites}-sites")


@dataclass
class AnalyticResult:
    model_name: str
    n_sites: int
    x_param: str
    y_param: str
    x_values: list
    y_values: list
    energies: np.ndarray
    log_path: str | None = None
    plot_path: str | None = None
    _model_params: dict = field(default_factory=dict, repr=False)

    def plot(self, *, hide_plot: bool = False, output_path=None, heatmap: bool = False):
        from quaph._registry import get_model
        model = get_model(self.model_name)

        x_label = f"${model.param_labels.get(self.x_param, self.x_param)}$" if self.x_param != "n_occ" else r"$N_{\text{occ}}$"
        y_label = f"${model.param_labels.get(self.y_param, self.y_param)}$" if self.y_param != "n_occ" else r"$N_{\text{occ}}$"

        return plot_analytic(
            self.x_values, self.y_values, x_label, y_label, self.energies,
            output_path=output_path, hide_plot=hide_plot, heatmap=heatmap,
        )


@dataclass
class SimulatedResult:
    model_name: str
    n_sites: int
    x_param: str
    y_param: str
    x_values: list
    y_values: list
    analytic_energies: np.ndarray
    vqe_best_energies: np.ndarray | None
    iqpe_best_energies: np.ndarray | None
    raw: dict = field(default_factory=dict, repr=False)
    raw_log_path: str | None = None
    summary_log_path: str | None = None
    plot_path: str | None = None
    _model_params: dict = field(default_factory=dict, repr=False)

    def plot(self, *, hide_plot: bool = False, output_path=None,
             hide_legend: bool = False):
        from quaph._registry import get_model
        model = get_model(self.model_name)

        x_label = f"${model.param_labels.get(self.x_param, self.x_param)}$" if self.x_param != "n_occ" else r"$N_{\text{occ}}$"
        y_label = f"${model.param_labels.get(self.y_param, self.y_param)}$" if self.y_param != "n_occ" else r"$N_{\text{occ}}$"

        return plot_simulated(
            self.x_values, self.y_values, x_label, y_label,
            self.analytic_energies, self.vqe_best_energies, self.iqpe_best_energies,
            hide_legend=hide_legend,
            output_path=output_path, hide_plot=hide_plot,
        )


def load_result(path: str) -> AnalyticResult | SimulatedResult:
    with open(path) as f:
        data = json.load(f)

    result_type = data.get("type", "")
    x_vals = data["x_values"]
    y_vals = data["y_values"]
    nx, ny = len(x_vals), len(y_vals)

    if result_type == "analytic":
        energies = np.array([[data["result"]["analytic"][str(ix)][str(iy)] for iy in range(ny)] for ix in range(nx)])
        return AnalyticResult(
            model_name=data["parameters"]["model"],
            n_sites=data["parameters"]["n_sites"],
            x_param=data["x_param"],
            y_param=data["y_param"],
            x_values=x_vals,
            y_values=y_vals,
            energies=energies,
            log_path=path,
            _model_params=data["parameters"].get("model_params", {}),
        )
    elif result_type in ("simulated-ideal", "simulated-noisy"):
        Z_exact = np.array([[data["result"]["analytic"][str(ix)][str(iy)] for iy in range(ny)] for ix in range(nx)])
        Z_vqe = (np.array([[data["result"]["vqe"][str(ix)][str(iy)] for iy in range(ny)] for ix in range(nx)])
                 if "vqe" in data["result"] else None)
        Z_iqpe = (np.array([[data["result"]["iqpe"][str(ix)][str(iy)] for iy in range(ny)] for ix in range(nx)])
                  if "iqpe" in data["result"] else None)
        return SimulatedResult(
            model_name=data["parameters"]["model"],
            n_sites=data["parameters"]["n_sites"],
            x_param=data["x_param"],
            y_param=data["y_param"],
            x_values=x_vals,
            y_values=y_vals,
            analytic_energies=Z_exact,
            vqe_best_energies=Z_vqe,
            iqpe_best_energies=Z_iqpe,
            raw=data,
            summary_log_path=path,
            _model_params=data["parameters"].get("model_params", {}),
        )
    else:
        raise ValueError(
            f"Unrecognized log file type '{result_type}' in {path}. "
            "Expected 'analytic', 'simulated-ideal', or 'simulated-noisy'."
        )


def run_analytic(
    model,
    *,
    n_sites: int,
    x_param: str | None = None,
    x_range=None,
    y_param: str | None = None,
    y_range=None,
    n_occ: int | None = None,
    model_params: dict | None = None,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    heatmap: bool = False,
) -> AnalyticResult:
    model = _resolve_model(model)
    _ = model._build_H_matrix

    x_param, x_range, y_param, y_range = _resolve_sweep_params(model, x_param, x_range, y_param, y_range)

    for axis in (x_param, y_param):
        if axis != "n_occ" and axis in (model_params or {}):
            raise ValueError(
                f"'{axis}' is the active sweep axis and cannot be set as a fixed value in model_params. "
                f"Override the sweep with x_param/y_param instead."
            )

    params = {**model.default_params, **(model_params or {})}
    fixed_n_occ = n_occ if n_occ is not None else n_sites
    spin = 2

    x_vals, x_label, x_is_nocc = resolve_sweep(x_param, x_range, n_sites, spin)
    y_vals, y_label, y_is_nocc = resolve_sweep(y_param, y_range, n_sites, spin)

    if not x_is_nocc:
        x_label = f"${model.param_labels.get(x_param, x_param)}$"
    if not y_is_nocc:
        y_label = f"${model.param_labels.get(y_param, y_param)}$"

    Z = np.full((len(x_vals), len(y_vals)), np.nan)

    for ix, xv in enumerate(x_vals):
        for iy, yv in enumerate(y_vals):
            cell_params = params.copy()
            n_occ_val = fixed_n_occ
            if x_is_nocc:
                n_occ_val = int(xv)
            else:
                cell_params[x_param] = xv
            if y_is_nocc:
                n_occ_val = int(yv)
            else:
                cell_params[y_param] = yv
            Z[ix, iy] = analytic(model, n_sites, n_occ_val, cell_params)

    log_path = None
    if log_dir is not None:
        subdir = _log_subdir(log_dir, model.name, n_sites)
        os.makedirs(subdir, exist_ok=True)
        log_path = os.path.join(subdir, f"analytic-{x_param}-vs-{y_param}.json")
        log_data = {
            "type": "analytic",
            "parameters": {
                "model": model.name,
                "n_sites": n_sites,
                "model_params": {k: float(v) for k, v in params.items()},
            },
            "x_param": x_param,
            "y_param": y_param,
            "x_values": x_vals,
            "y_values": y_vals,
            "result": {
                "analytic": {ix: {iy: Z[ix, iy] for iy in range(len(y_vals))} for ix in range(len(x_vals))}
            },
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        subdir = _plot_subdir(plot_dir, model.name, n_sites)
        os.makedirs(subdir, exist_ok=True)
        plot_path = os.path.join(subdir, f"analytic-{x_param}-vs-{y_param}.pdf")

    if plot_path is not None or not hide_plot:
        plot_analytic(
            x_vals, y_vals, x_label, y_label, Z,
            output_path=plot_path, hide_plot=hide_plot, heatmap=heatmap,
        )

    return AnalyticResult(
        model_name=model.name,
        n_sites=n_sites,
        x_param=x_param,
        y_param=y_param,
        x_values=x_vals,
        y_values=y_vals,
        energies=Z,
        log_path=log_path,
        plot_path=plot_path,
        _model_params=params,
    )


def _run_simulated(
    model: Model,
    simulation_tag: str,
    backend,
    *,
    n_sites: int,
    x_param: str,
    x_range,
    y_param: str,
    y_range,
    n_occ: int | None,
    model_params: dict,
    vqe_iters: int | None,
    vqe_layers: int | None,
    vqe_reps: int,
    iqpe_time: float | None,
    iqpe_trot: int | None,
    iqpe_iters: int | None,
    iqpe_reps: int,
    log_dir,
    plot_dir,
    hide_plot: bool,
    hide_legend: bool,
) -> SimulatedResult:
    do_vqe = vqe_reps > 0
    do_iqpe = iqpe_reps > 0
    _ = model.fermionic_hamiltonian

    spin = 2
    mapper = JordanWignerMapper()
    fixed_n_occ = n_occ if n_occ is not None else n_sites

    x_vals, x_label, x_is_nocc = resolve_sweep(x_param, x_range, n_sites, spin)
    y_vals, y_label, y_is_nocc = resolve_sweep(y_param, y_range, n_sites, spin)

    if not x_is_nocc:
        x_label = f"${model.param_labels.get(x_param, x_param)}$"
    if not y_is_nocc:
        y_label = f"${model.param_labels.get(y_param, y_param)}$"

    def cell_params_and_nocc(ix, iy):
        cp = model_params.copy()
        n_occ_val = fixed_n_occ
        xv, yv = x_vals[ix], y_vals[iy]
        if x_is_nocc:
            n_occ_val = int(xv)
        else:
            cp[x_param] = xv
        if y_is_nocc:
            n_occ_val = int(yv)
        else:
            cp[y_param] = yv
        return cp, n_occ_val

    def tagged_job(tag, func, *a, **kw):
        return tag, func(*a, **kw)

    jobs = []
    for ix in range(len(x_vals)):
        for iy in range(len(y_vals)):
            cp, n_occ_val = cell_params_and_nocc(ix, iy)
            jobs.append(delayed(tagged_job)(
                ("analytic", ix, iy), analytic, model, n_sites, n_occ_val, cp
            ))
            if do_iqpe:
                for rep in range(1, iqpe_reps + 1):
                    jobs.append(delayed(tagged_job)(
                        ("iqpe", ix, iy, rep), iqpe,
                        n_sites, n_occ_val, cp, model.fermionic_hamiltonian,
                        mapper, iqpe_time, iqpe_trot, iqpe_iters, rep,
                        backend=backend
                    ))
                jobs.append(delayed(tagged_job)(
                    ("iqpe_bench", ix, iy), iqpe_other_benchmarks,
                    n_sites, n_occ_val, cp, model.fermionic_hamiltonian,
                    mapper, iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps,
                    backend=backend
                ))
            if do_vqe:
                for rep in range(1, vqe_reps + 1):
                    jobs.append(delayed(tagged_job)(
                        ("vqe", ix, iy, rep), vqe,
                        n_sites, n_occ_val, cp, model.fermionic_hamiltonian, model.get_optimizer,
                        mapper, vqe_iters, vqe_layers, rep,
                        backend=backend
                    ))
                jobs.append(delayed(tagged_job)(
                    ("vqe_bench", ix, iy), vqe_other_benchmarks,
                    n_sites, n_occ_val, cp, model.fermionic_hamiltonian,
                    mapper, vqe_iters, vqe_layers, vqe_reps,
                    backend=backend
                ))

    raw_data_path = None
    if log_dir is not None:
        log_subdir = _log_subdir(log_dir, model.name, n_sites)
        os.makedirs(os.path.join(log_subdir, "raw-data"), exist_ok=True)
        raw_data_path = os.path.join(log_subdir, "raw-data", f"simulated-{simulation_tag}-{x_param}-vs-{y_param}.json")

    def empty_cell():
        cell = {"analytic": None}
        if do_vqe:
            cell["vqe"] = {"repetitions": [], "num_queries": None, "circuit_depth": None}
        if do_iqpe:
            cell["iqpe"] = {"repetitions": [], "iteration_energies": [], "num_queries": None, "circuit_depth": None}
        return cell

    parameters = {
        "model": model.name,
        "n_sites": n_sites,
        "simulation": simulation_tag,
        "model_params": {k: float(v) for k, v in model_params.items()},
    }
    if do_vqe:
        parameters["vqe"] = {"iters": vqe_iters, "layers": vqe_layers, "reps": vqe_reps}
    if do_iqpe:
        parameters["iqpe"] = {"time": iqpe_time, "trot": iqpe_trot, "iters": iqpe_iters, "reps": iqpe_reps}

    raw_data = {
        "parameters": parameters,
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "grid": {
            str(ix): {str(iy): empty_cell() for iy in range(len(y_vals))}
            for ix in range(len(x_vals))
        },
    }

    if raw_data_path is not None:
        with open(raw_data_path, "w") as f:
            json.dump(raw_data, f, indent=4)

    def init_worker_logging():
        from quaph._core import setup_logging as _sl
        _sl()

    for tag, result in Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(jobs):
        ix, iy = str(tag[1]), str(tag[2])
        cell = raw_data["grid"][ix][iy]
        if tag[0] == "analytic":
            cell["analytic"] = result
        elif tag[0] == "iqpe":
            energy, iter_energies = result
            cell["iqpe"]["repetitions"].append(energy)
            cell["iqpe"]["iteration_energies"].append(iter_energies)
        elif tag[0] == "vqe":
            cell["vqe"]["repetitions"].append(result)
        elif tag[0] == "iqpe_bench":
            num_q, (total, two_q) = result
            cell["iqpe"]["num_queries"] = num_q
            cell["iqpe"]["circuit_depth"] = {"total": total, "two_qubit": two_q}
        elif tag[0] == "vqe_bench":
            num_q, (total, two_q) = result
            cell["vqe"]["num_queries"] = num_q
            cell["vqe"]["circuit_depth"] = {"total": total, "two_qubit": two_q}
        if raw_data_path is not None:
            with open(raw_data_path, "w") as f:
                json.dump(raw_data, f, indent=4)

    from loguru import logger

    nx, ny = len(x_vals), len(y_vals)
    Z_exact = np.full((nx, ny), np.nan)
    Z_vqe = np.full((nx, ny), np.nan) if do_vqe else None
    Z_iqpe = np.full((nx, ny), np.nan) if do_iqpe else None

    for ix in range(nx):
        for iy in range(ny):
            cell = raw_data["grid"][str(ix)][str(iy)]
            Z_exact[ix, iy] = cell["analytic"]
            if do_iqpe:
                Z_iqpe[ix, iy] = min(cell["iqpe"]["repetitions"], key=lambda e: abs(e - Z_exact[ix, iy]))
                logger.info(f"IQPE ({x_param}={x_vals[ix]}, {y_param}={y_vals[iy]}) = {Z_iqpe[ix, iy]}")
            if do_vqe:
                Z_vqe[ix, iy] = min(cell["vqe"]["repetitions"], key=lambda e: abs(e - Z_exact[ix, iy]))
                logger.info(f"VQE  ({x_param}={x_vals[ix]}, {y_param}={y_vals[iy]}) = {Z_vqe[ix, iy]}")

    result_block = {"analytic": {ix: {iy: Z_exact[ix, iy] for iy in range(ny)} for ix in range(nx)}}
    num_queries_block = {}
    depth_total_block = {}
    depth_two_q_block = {}
    if do_iqpe:
        result_block["iqpe"] = {ix: {iy: Z_iqpe[ix, iy] for iy in range(ny)} for ix in range(nx)}
        num_queries_block["iqpe"] = {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["iqpe"]["num_queries"] for iy in range(ny)} for ix in range(nx)}
        depth_total_block["iqpe"] = {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["iqpe"]["circuit_depth"]["total"] for iy in range(ny)} for ix in range(nx)}
        depth_two_q_block["iqpe"] = {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["iqpe"]["circuit_depth"]["two_qubit"] for iy in range(ny)} for ix in range(nx)}
    if do_vqe:
        result_block["vqe"] = {ix: {iy: Z_vqe[ix, iy] for iy in range(ny)} for ix in range(nx)}
        num_queries_block["vqe"] = {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["vqe"]["num_queries"] for iy in range(ny)} for ix in range(nx)}
        depth_total_block["vqe"] = {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["vqe"]["circuit_depth"]["total"] for iy in range(ny)} for ix in range(nx)}
        depth_two_q_block["vqe"] = {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["vqe"]["circuit_depth"]["two_qubit"] for iy in range(ny)} for ix in range(nx)}

    summary = {
        "type": f"simulated-{simulation_tag}",
        "parameters": raw_data["parameters"],
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "result": result_block,
    }
    if do_vqe or do_iqpe:
        summary["num_queries"] = num_queries_block
        summary["circuit_depth"] = {"total": depth_total_block, "two_qubit": depth_two_q_block}

    summary_path = None
    if log_dir is not None:
        summary_path = os.path.join(log_subdir, f"simulated-{simulation_tag}-{x_param}-vs-{y_param}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        plot_subdir = _plot_subdir(plot_dir, model.name, n_sites)
        os.makedirs(plot_subdir, exist_ok=True)
        plot_path = os.path.join(plot_subdir, f"simulated-{simulation_tag}-{x_param}-vs-{y_param}.pdf")

    if plot_path is not None or not hide_plot:
        plot_simulated(
            x_vals, y_vals, x_label, y_label, Z_exact, Z_vqe, Z_iqpe,
            hide_legend=hide_legend,
            output_path=plot_path, hide_plot=hide_plot,
        )

    return SimulatedResult(
        model_name=model.name,
        n_sites=n_sites,
        x_param=x_param,
        y_param=y_param,
        x_values=x_vals,
        y_values=y_vals,
        analytic_energies=Z_exact,
        vqe_best_energies=Z_vqe,
        iqpe_best_energies=Z_iqpe,
        raw=raw_data,
        raw_log_path=raw_data_path,
        summary_log_path=summary_path,
        plot_path=plot_path,
        _model_params=model_params,
    )


def _resolve_method_reps(name, reps, *params):
    any_param = any(p is not None for p in params)
    if reps is None:
        return 1 if any_param else 0
    if reps > 0 and not any_param:
        raise ValueError(f"{name}_reps={reps} but no {name} parameters provided.")
    return reps


def run_simulated_ideal(
    model,
    *,
    n_sites: int,
    x_param: str | None = None,
    x_range=None,
    y_param: str | None = None,
    y_range=None,
    n_occ: int | None = None,
    model_params: dict | None = None,
    vqe_iters: int | None = None,
    vqe_layers: int | None = None,
    vqe_reps: int | None = None,
    iqpe_time: float | None = None,
    iqpe_trot: int | None = None,
    iqpe_iters: int | None = None,
    iqpe_reps: int | None = None,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
) -> SimulatedResult:
    model = _resolve_model(model)
    vqe_reps = _resolve_method_reps("vqe", vqe_reps, vqe_iters, vqe_layers)
    iqpe_reps = _resolve_method_reps("iqpe", iqpe_reps, iqpe_time, iqpe_trot, iqpe_iters)
    x_param, x_range, y_param, y_range = _resolve_sweep_params(model, x_param, x_range, y_param, y_range)

    for axis in (x_param, y_param):
        if axis != "n_occ" and axis in (model_params or {}):
            raise ValueError(
                f"'{axis}' is the active sweep axis and cannot be set as a fixed value in model_params. "
                f"Override the sweep with x_param/y_param instead."
            )

    params = {**model.default_params, **(model_params or {})}

    return _run_simulated(
        model, "ideal", None,
        n_sites=n_sites, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ,
        model_params=params, vqe_iters=vqe_iters, vqe_layers=vqe_layers,
        vqe_reps=vqe_reps, iqpe_time=iqpe_time, iqpe_trot=iqpe_trot,
        iqpe_iters=iqpe_iters, iqpe_reps=iqpe_reps,
        log_dir=log_dir, plot_dir=plot_dir,
        hide_plot=hide_plot, hide_legend=hide_legend,
    )


def run_simulated_noisy(
    model,
    *,
    backend=None,
    n_sites: int,
    x_param: str | None = None,
    x_range=None,
    y_param: str | None = None,
    y_range=None,
    n_occ: int | None = None,
    model_params: dict | None = None,
    vqe_iters: int | None = None,
    vqe_layers: int | None = None,
    vqe_reps: int | None = None,
    iqpe_time: float | None = None,
    iqpe_trot: int | None = None,
    iqpe_iters: int | None = None,
    iqpe_reps: int | None = None,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
) -> SimulatedResult:
    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()

    model = _resolve_model(model)
    vqe_reps = _resolve_method_reps("vqe", vqe_reps, vqe_iters, vqe_layers)
    iqpe_reps = _resolve_method_reps("iqpe", iqpe_reps, iqpe_time, iqpe_trot, iqpe_iters)
    x_param, x_range, y_param, y_range = _resolve_sweep_params(model, x_param, x_range, y_param, y_range)

    for axis in (x_param, y_param):
        if axis != "n_occ" and axis in (model_params or {}):
            raise ValueError(
                f"'{axis}' is the active sweep axis and cannot be set as a fixed value in model_params. "
                f"Override the sweep with x_param/y_param instead."
            )

    params = {**model.default_params, **(model_params or {})}

    return _run_simulated(
        model, "noisy", backend,
        n_sites=n_sites, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ,
        model_params=params, vqe_iters=vqe_iters, vqe_layers=vqe_layers,
        vqe_reps=vqe_reps, iqpe_time=iqpe_time, iqpe_trot=iqpe_trot,
        iqpe_iters=iqpe_iters, iqpe_reps=iqpe_reps,
        log_dir=log_dir, plot_dir=plot_dir,
        hide_plot=hide_plot, hide_legend=hide_legend,
    )
