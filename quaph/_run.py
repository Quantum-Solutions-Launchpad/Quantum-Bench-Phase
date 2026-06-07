from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from quaph._model import Model, ModelCapabilityError
from quaph._core import resolve_sweep
from quaph._hamlib import (
    list_hamlib_keys, load_hamlib_operator, parse_key_params,
)
from quaph._plotting import plot_analytic, plot_simulated
from quaph._registry import get_model as _get_model
from quaph._method import (
    Method, METHOD_ORDER, CellContext, build_method, get_method_class,
)


# Marker styling for non-surface comparison series (VQE/IQPE handled natively by
# plot_simulated; everything else is rendered through extra_series).
_METHOD_STYLE = {
    "dmrg": {"color": "#D55E00", "marker": "D"},
    "vqe": {"color": "#0072B2", "marker": "o"},
    "iqpe": {"color": "#6DBF82", "marker": "^"},
    "analytic": {"color": "#888888", "marker": "s"},
}


def _resolve_model(model) -> Model:
    if isinstance(model, str):
        return _get_model(model)
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance or model name string, got {type(model).__name__}")
    return model


def _resolve_lattice(model: Model, lattice) -> tuple[int, ...]:
    if lattice is None:
        raise ValueError(
            f"Model '{model.name}' requires lattice={model.lattice_shape} "
            f"(n_dims={model.n_dims}, sites_per_cell={model.sites_per_cell})."
        )
    try:
        lat = tuple(int(x) for x in lattice)
    except TypeError:
        raise ValueError(
            f"lattice must be a sequence of ints matching {model.lattice_shape}; got {lattice!r}"
        )
    if len(lat) != model.n_dims:
        raise ValueError(
            f"Model '{model.name}' expects lattice with {model.n_dims} entries "
            f"(shape={model.lattice_shape}); got lattice={lat}."
        )
    if any(x < 1 for x in lat):
        raise ValueError(f"lattice entries must be >= 1; got {lat}.")
    return lat


def _is_band_structure_axes(model, x_param, y_param) -> bool:
    return (x_param in model.momentum_axes) or (y_param in model.momentum_axes)


def _normalize_sweep_axes(x_param, x_range, y_param, y_range):
    if x_param is None and y_param is None:
        raise ValueError("At least one of x_param / y_param must be provided.")
    if x_param is None:
        x_param, y_param = y_param, None
        x_range, y_range = y_range, None
    is_1d = y_param is None
    if x_param != "n_occ" and x_range is None:
        raise ValueError(f"x_range is required for sweep axis '{x_param}'.")
    if not is_1d and y_param != "n_occ" and y_range is None:
        raise ValueError(f"y_range is required for sweep axis '{y_param}'.")
    return x_param, x_range, y_param, y_range, is_1d


def _label_for(model, param: str) -> str:
    if param == "n_occ":
        return r"$N_{\text{occ}}$"
    return f"${model.param_labels.get(param, param)}$"


def _opt_lattice(lat):
    return tuple(lat) if lat else None


def _observable_label(model, observable: str) -> str:
    obs = model.get_observable(observable)
    return f"${obs.display_name}$"


def _result_labels(model_name, x_param, y_param):
    from quaph._registry import get_model
    try:
        model = get_model(model_name)
    except Exception:
        x_label = "Instance" if x_param == "instance" else f"${x_param}$"
        y_label = f"${y_param}$" if y_param else "$E$"
        return x_label, y_label, False, False
    x_label = _label_for(model, x_param)
    y_label = _label_for(model, y_param) if y_param else "$E$"
    x_is_momentum = x_param in model.momentum_axes
    y_is_momentum = bool(y_param) and y_param in model.momentum_axes
    return x_label, y_label, x_is_momentum, y_is_momentum


def _safe_observable_label(model_name, observable: str) -> str:
    from quaph._registry import get_model
    try:
        return f"${get_model(model_name).get_observable(observable).display_name}$"
    except Exception:
        return "$E$"


def _derived_paths(log_path):
    """Sidecar raw + progress journal paths derived from a user's log_path."""
    base = os.path.splitext(log_path)[0]
    return f"{base}.raw-data.json", f"{base}.progress.jsonl"


def _gate_momentum(model, x_param, y_param):
    all_momentum_names = {"k", "kx", "ky", "kz"}
    for axis in (x_param, y_param):
        if axis in all_momentum_names and axis not in model.momentum_axes:
            raise ModelCapabilityError(
                f"Model '{model.name}' (n_dims={model.n_dims}) does not have momentum axis '{axis}'; "
                f"available momentum axes are {model.momentum_axes}."
            )
        if axis in all_momentum_names and not model.supports_band_structure:
            raise ModelCapabilityError(
                f"Model '{model.name}' does not implement bloch_hamiltonian; "
                f"momentum-space (band structure) runs along '{axis}' are not supported."
            )


# ----------------------------------------------------------------- method setup
def _normalize_methods(method) -> list[Method]:
    if method is None:
        raise ValueError("run() requires method=<Method or list of Methods>.")
    if isinstance(method, (Method, str)):
        items = [method]
    else:
        items = list(method)
    if not items:
        raise ValueError("run() requires at least one simulation method.")
    seen = {Method.coerce(m) for m in items}
    return [m for m in METHOD_ORDER if m in seen]


def _build_method_objects(methods, method_params):
    method_params = method_params or {}
    objs = {}
    for m in methods:
        params = method_params.get(m)
        if params is None:
            params = method_params.get(m.value, {})
        objs[m] = build_method(m, params)
    return objs


# ----------------------------------------------------------------------- result
@dataclass
class AnalyticResult:
    model_name: str
    lattice: tuple[int, ...] | None
    x_param: str
    y_param: str | None
    x_values: list
    y_values: list
    methods: list[str]
    grids: dict[str, np.ndarray]
    band_structure: bool = False
    analytic_bands: np.ndarray | None = None
    plot_format: str = "3d"
    observable: str = "E"
    extremum: str = "min"
    backend_label: str = "ideal"
    log_path: str | None = None
    raw_log_path: str | None = None
    plot_path: str | None = None
    raw: dict = field(default_factory=dict, repr=False)
    _model_params: dict = field(default_factory=dict, repr=False)

    def plot(self, *, hide_plot: bool = False, output_path=None):
        from quaph._registry import get_model
        model = get_model(self.model_name)
        x_label = _label_for(model, self.x_param)
        y_label = _label_for(model, self.y_param) if self.y_param else "$E$"
        x_is_momentum = self.x_param in model.momentum_axes
        y_is_momentum = bool(self.y_param) and self.y_param in model.momentum_axes
        return plot_analytic(
            rr.x_values, rr.y_values, x_label, y_label if rr.y_param else z_label, Z,
            plot_format=plot_format, output_path=output_path, hide_plot=hide_plot,
            x_is_momentum=x_is_mom, y_is_momentum=y_is_mom, z_label=z_label,
        )


@dataclass
class SimulatedResult:
    model_name: str
    lattice: tuple[int, ...]
    x_param: str
    y_param: str | None
    x_values: list
    y_values: list
    analytic_energies: np.ndarray
    vqe_best_energies: np.ndarray | None
    iqpe_best_energies: np.ndarray | None
    plot_format: str = "3d"
    band_structure: bool = False
    analytic_bands: np.ndarray | None = None
    raw: dict = field(default_factory=dict, repr=False)
    raw_log_path: str | None = None
    summary_log_path: str | None = None
    plot_path: str | None = None
    _model_params: dict = field(default_factory=dict, repr=False)

    def plot(self, *, hide_plot: bool = False, output_path=None,
             hide_legend: bool = False):
        from quaph._registry import get_model
        model = get_model(self.model_name)
        x_label = _label_for(model, self.x_param)
        y_label = _label_for(model, self.y_param) if self.y_param else "$E$"
        x_is_momentum = self.x_param in model.momentum_axes
        y_is_momentum = bool(self.y_param) and self.y_param in model.momentum_axes
        Z_exact = self.analytic_bands if self.band_structure else self.analytic_energies
        return plot_simulated(
            self.x_values, self.y_values, x_label, y_label,
            Z_exact, self.vqe_best_energies, self.iqpe_best_energies,
            plot_format=self.plot_format,
            hide_legend=hide_legend,
            output_path=output_path, hide_plot=hide_plot,
            x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
        )


def load_result(path: str) -> AnalyticResult | SimulatedResult:
    with open(path) as f:
        data = json.load(f)

    if data.get("type") != "run":
        raise ValueError(
            f"Unrecognized log file type {data.get('type')!r} in {path}. "
            "Expected a unified 'run' log (regenerate with the current API)."
        )

    x_vals = data["x_values"]
    y_vals = data.get("y_values", []) or []
    is_1d = len(y_vals) == 0
    nx = len(x_vals)
    ny = len(y_vals) if not is_1d else 1
    band_structure = bool(data.get("band_structure", False))
    plot_format = data.get("plot_format", "2d" if is_1d else "3d")
    methods = list(data["methods"])

    def _scalar_grid(block):
        arr = np.full((nx, ny), np.nan)
        for ix in range(nx):
            col = block[str(ix)]
            if is_1d:
                arr[ix, 0] = col
            else:
                for iy in range(ny):
                    arr[ix, iy] = col[str(iy)]
        return _squeeze_scalar(arr, is_1d)

    def _bands_grid(block):
        probe = block["0"] if is_1d else block["0"]["0"]
        nb = len(probe)
        arr = np.full((nx, ny, nb), np.nan)
        for ix in range(nx):
            col = block[str(ix)]
            if is_1d:
                arr[ix, 0, :] = col
            else:
                for iy in range(ny):
                    arr[ix, iy, :] = col[str(iy)]
        return _squeeze_bands(arr, is_1d)

    grids = {}
    analytic_bands = None
    for m in methods:
        block = data["result"][m]
        if band_structure and m == "analytic":
            analytic_bands = _bands_grid(block)
            grids[m] = analytic_bands[..., 0]
        else:
            grids[m] = _scalar_grid(block)

    params = data.get("parameters", {})
    return RunResult(
        model_name=params.get("model"),
        lattice=_opt_lattice(params.get("lattice")),
        x_param=data["x_param"],
        y_param=data.get("y_param"),
        x_values=x_vals,
        y_values=y_vals if not is_1d else [],
        methods=methods,
        grids=grids,
        band_structure=band_structure,
        analytic_bands=analytic_bands,
        plot_format=plot_format,
        observable=data.get("observable", "E"),
        extremum=data.get("extremum", "min"),
        backend_label=data.get("backend", "ideal"),
        log_path=path,
        raw=data,
        _model_params=params.get("model_params", {}),
    )


# ----------------------------------------------------------- public entry point
def run(
    model=None,
    *,
    method,
    method_params: dict | None = None,
    lattice=None,
    x_param: str | None = None,
    x_range=None,
    y_param: str | None = None,
    y_range=None,
    n_occ: int | None = None,
    model_params: dict | None = None,
    observable: str = "E",
    backend=None,
    qubit_operator: str | None = None,
    extremum: str = "min",
    select=None,
    log_path=None,
    plot_path=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
    heatmap: bool = False,
) -> AnalyticResult:
    model = _resolve_model(model)
    _ = model._build_H_matrix
    _ = model.get_observable(observable)

    x_param, x_range, y_param, y_range, is_1d = _normalize_sweep_axes(
        x_param, x_range, y_param, y_range
    )
    if heatmap and is_1d:
        raise ValueError("heatmap=True requires both x and y sweep axes; provide y_param/y_range.")
    if heatmap and len(methods) != 1:
        raise ValueError("heatmap=True requires exactly one simulation method.")
    _gate_momentum(model, x_param, y_param)

    is_band_structure_run = _is_band_structure_axes(model, x_param, y_param)
    if is_band_structure_run:
        if lattice is not None:
            raise ValueError("lattice and momentum-space sweep axes are mutually exclusive; omit lattice for band-structure runs.")
        for m in methods:
            if not method_objs[m].SUPPORTS_BAND_STRUCTURE:
                raise ValueError(
                    f"Method '{m.value}' does not support band-structure (momentum-space) runs."
                )
    else:
        lattice = _resolve_lattice(model, lattice)

    for axis in (x_param, y_param):
        if axis is None:
            continue
        if axis != "n_occ" and axis in (model_params or {}):
            raise ValueError(
                f"'{axis}' is the active sweep axis and cannot be set as a fixed value in model_params. "
                f"Override the sweep with x_param/y_param instead."
            )

    if Method.IQPE in methods and observable != "E":
        from quaph._iqpe import iqpe_supports_observable
        if not iqpe_supports_observable(model, observable):
            raise ValueError(
                f"IQPE cannot measure observable '{observable}'; only 'E' and energy "
                f"composites (e.g. charge_gap) are supported. Drop IQPE or use VQE."
            )

    params = dict(model_params or {})
    spin = model.spin
    if lattice is not None:
        n_sites = math.prod(lattice) * model.sites_per_cell
        n_orbitals = n_sites * spin
    else:
        n_sites = 0
        n_orbitals = 0
    fixed_n_occ = n_occ if n_occ is not None else (n_orbitals // 2 if n_orbitals else 0)
    momentum_axes = model.momentum_axes

    x_vals, _x_label_default, x_kind = resolve_sweep(x_param, x_range, n_orbitals, momentum_axes)
    if is_1d:
        y_vals, y_kind = [], "none"
    else:
        y_vals, _y_label_default, y_kind = resolve_sweep(y_param, y_range, n_orbitals, momentum_axes)
    nx = len(x_vals)
    ny = 1 if is_1d else len(y_vals)

    is_band = (x_kind == "momentum") or (y_kind == "momentum")
    if is_band:
        if x_kind == "n_occ" or y_kind == "n_occ" or n_occ is not None:
            raise ValueError("n_occ is not meaningful for band-structure (momentum-space) runs.")
        _ = model.bloch_hamiltonian
        for k_axis in momentum_axes:
            if k_axis not in (x_param, y_param) and k_axis not in params:
                raise ValueError(
                    f"Model '{model.name}' has momentum axis '{k_axis}' that is neither the active "
                    f"sweep axis nor fixed in model_params. Pin it explicitly to a value."
                )

    plot_format = "2d" if is_1d else ("heatmap" if heatmap else "3d")

    if task_count < 1:
        raise ValueError("task_count must be at least 1")
    if task_index is not None and not 0 <= task_index < task_count:
        raise ValueError("task_index must satisfy 0 <= task_index < task_count")

    use_parallel = any(method_objs[m].WANTS_PARALLEL for m in methods)

    raw_data_path = None
    progress_path = None
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        # Raw sidecar + progress journal only matter for expensive parallel runs
        # (benchmarks, resume, sharding); analytic-only runs need just the summary.
        if use_parallel:
            raw_data_path, progress_path = _derived_paths(log_path)

    raw_cells = {m.value: {str(ix): {} for ix in range(nx)} for m in methods}

    def cell_params_and_nocc(ix, iy):
        cp = params.copy()
        n_occ_val = fixed_n_occ
        xv = x_vals[ix]
        if x_kind == "n_occ":
            n_occ_val = int(xv)
        else:
            cp[x_param] = xv
        if not is_1d:
            yv = y_vals[iy]
            if y_kind == "n_occ":
                n_occ_val = int(yv)
            else:
                cp[y_param] = yv
        return cp, n_occ_val

    def compute_one(method_value, ix, iy, tmp_dir):
        m_obj = method_objs[Method.coerce(method_value)]
        cp, n_occ_val = cell_params_and_nocc(ix, iy)
        label = f"{x_param}={x_vals[ix]}" + ("" if is_1d else f", {y_param}={y_vals[iy]}")
        ctx = CellContext(
            ix=ix, iy=iy, cell_index=ix * ny + iy,
            n_sites=n_sites, spin=spin, n_orbitals=n_orbitals,
            raw_dir=tmp_dir, tmp_dir=tmp_dir, label=label,
        )
        if is_band:
            k_tuple = tuple(cp.pop(a) for a in momentum_axes)
            return m_obj.compute_bloch_cell(model, k_tuple, cp, observable, backend=backend, ctx=ctx)
        ctx.mapper = model.get_mapper(n_sites, spin, n_occ_val)
        return m_obj.compute_cell(model, lattice, n_occ_val, cp, observable, backend=backend, ctx=ctx)

    def run_job(tag_tuple, tmp_dir):
        return tag_tuple, compute_one(*tag_tuple, tmp_dir=tmp_dir)

    all_tags = [(m.value, ix, iy) for m in methods for ix in range(nx) for iy in range(ny)]

    def apply(tag_tuple, cell):
        mv, ix, iy = tag_tuple
        raw_cells[mv][str(ix)][str(iy)] = cell

    def append_progress(tag_tuple, cell):
        if no_progress_log or progress_path is None:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tag": list(tag_tuple),
            "cell": cell,
        }
        payload = (json.dumps(record) + "\n").encode()
        fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    def load_progress():
        if progress_path is None:
            raise ValueError("log_path is required for aggregate_only")
        if not os.path.exists(progress_path):
            raise FileNotFoundError(f"Progress file does not exist: {progress_path}")
        with open(progress_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                apply(tuple(record["tag"]), record["cell"])

    def validate_complete():
        missing = []
        for m in methods:
            for ix in range(nx):
                for iy in range(ny):
                    if str(iy) not in raw_cells[m.value].get(str(ix), {}):
                        missing.append(f"{m.value}:{ix},{iy}")
        if missing:
            raise RuntimeError("Missing results before aggregation: " + ", ".join(missing[:20]))

    method_params_summary = {m.value: method_objs[m].parameter_summary() for m in methods}

    def build_raw():
        return {
            "type": "run",
            "methods": [m.value for m in methods],
            "backend": backend_label,
            "plot_format": plot_format,
            "band_structure": is_band,
            "observable": observable,
            "extremum": "min",
            "x_param": x_param, "y_param": y_param,
            "x_values": x_vals, "y_values": y_vals,
            "parameters": {
                "model": model.name,
                "lattice": list(lattice) if lattice is not None else None,
                "model_params": {k: float(v) for k, v in params.items()},
                "method_params": method_params_summary,
            },
            "cells": raw_cells,
        }

    if raw_data_path is not None and not no_progress_log and (
        prepare_only or (task_index is None and not aggregate_only)
    ):
        with open(progress_path, "w") as f:
            f.write("")

    def empty_result():
        grids = {m.value: _squeeze_scalar(np.full((nx, ny), np.nan), is_1d) for m in methods}
        return RunResult(
            model_name=model.name, lattice=lattice, x_param=x_param, y_param=y_param,
            x_values=x_vals, y_values=y_vals if not is_1d else [],
            methods=[m.value for m in methods], grids=grids,
            band_structure=is_band, plot_format=plot_format, observable=observable,
            backend_label=backend_label, raw=build_raw(), raw_log_path=raw_data_path,
            _model_params=params,
        )

    if prepare_only:
        if raw_data_path is not None:
            with open(raw_data_path, "w") as f:
                json.dump(build_raw(), f, indent=4)
        return empty_result()

    def init_worker_logging():
        from quaph._core import setup_logging as _sl
        _sl()

    def jobs_per_shard():
        value = os.environ.get("QUAPH_JOBS_PER_SHARD") or "1"
        try:
            return max(1, int(value))
        except ValueError:
            return 1

    if aggregate_only:
        load_progress()
    else:
        if task_index is None:
            selected = all_tags
            n_jobs = -1
        else:
            init_worker_logging()
            selected = [t for i, t in enumerate(all_tags) if i % task_count == task_index]
            n_jobs = jobs_per_shard()

        with tempfile.TemporaryDirectory(prefix="quaph-run-") as tmp_dir:
            if use_parallel:
                results = Parallel(
                    n_jobs=n_jobs, return_as="generator_unordered",
                    initializer=init_worker_logging,
                )(delayed(run_job)(t, tmp_dir) for t in selected)
            else:
                # Cheap cells (analytic): joblib's per-task pickling overhead
                # dominates, so run in-process.
                results = (run_job(t, tmp_dir) for t in selected)
            for tag_tuple, cell in results:
                if use_parallel:
                    append_progress(tag_tuple, cell)
                apply(tag_tuple, cell)
                # Incremental snapshots only for expensive parallel runs; for
                # large cheap grids a per-cell full dump would be O(n^2).
                if use_parallel and raw_data_path is not None and task_index is None:
                    with open(raw_data_path, "w") as f:
                        json.dump(build_raw(), f, indent=4)

        if task_index is not None:
            return empty_result()

    validate_complete()

    # --------------------------------------------------------------- reduce grids
    n_bands = 1
    analytic_bands = None
    grids_full = {}
    for m in methods:
        m_obj = method_objs[m]
        if is_band and m == Method.ANALYTIC:
            probe = raw_cells["analytic"]["0"]["0"]["bands"]
            n_bands = len(probe)
            arr = np.full((nx, ny, n_bands), np.nan)
            for ix in range(nx):
                for iy in range(ny):
                    arr[ix, iy, :] = m_obj.reduce(raw_cells["analytic"][str(ix)][str(iy)])
            analytic_bands = arr
            grids_full[m.value] = arr[..., 0]
        else:
            arr = np.full((nx, ny), np.nan)
            for ix in range(nx):
                for iy in range(ny):
                    val = m_obj.reduce(raw_cells[m.value][str(ix)][str(iy)], extremum="min")
                    arr[ix, iy] = val
            grids_full[m.value] = arr
            for ix in range(nx):
                for iy in range(ny):
                    loc = f"{x_param}={x_vals[ix]}" + ("" if is_1d else f", {y_param}={y_vals[iy]}")
                    logger.info(f"{m_obj.LABEL} ({loc}) = {arr[ix, iy]}")

    # --------------------------------------------------------------- result block
    def scalar_block(arr):
        if is_1d:
            return {ix: float(arr[ix, 0]) for ix in range(nx)}
        return {ix: {iy: float(arr[ix, iy]) for iy in range(ny)} for ix in range(nx)}

    def bands_block(arr):
        if is_1d:
            return {ix: arr[ix, 0, :].tolist() for ix in range(nx)}
        return {ix: {iy: arr[ix, iy, :].tolist() for iy in range(ny)} for ix in range(nx)}

    result_block = {}
    for m in methods:
        if is_band and m == Method.ANALYTIC:
            result_block[m.value] = bands_block(analytic_bands)
        else:
            result_block[m.value] = scalar_block(grids_full[m.value])

    summary = {
        "type": "run",
        "methods": [m.value for m in methods],
        "backend": backend_label,
        "plot_format": plot_format,
        "band_structure": is_band,
        "observable": observable,
        "extremum": "min",
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "parameters": {
            "model": model.name,
            "lattice": list(lattice) if lattice is not None else None,
            "model_params": {k: float(v) for k, v in params.items()},
            "method_params": method_params_summary,
        },
        "result": result_block,
    }
    if is_band:
        summary["n_bands"] = n_bands

    if raw_data_path is not None:
        with open(raw_data_path, "w") as f:
            json.dump(build_raw(), f, indent=4)

    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=4)

    # squeeze grids for the result object
    grids_out = {}
    for m in methods:
        if is_band and m == Method.ANALYTIC:
            grids_out[m.value] = _squeeze_bands(analytic_bands, is_1d)[..., 0] if is_1d else analytic_bands[..., 0]
        else:
            grids_out[m.value] = _squeeze_scalar(grids_full[m.value], is_1d)
    analytic_bands_out = _squeeze_bands(analytic_bands, is_1d) if analytic_bands is not None else None

    result = RunResult(
        model_name=model.name, lattice=lattice, x_param=x_param, y_param=y_param,
        x_values=x_vals, y_values=y_vals if not is_1d else [],
        methods=[m.value for m in methods], grids=grids_out,
        band_structure=is_band, analytic_bands=analytic_bands_out,
        plot_format=plot_format, observable=observable, extremum="min",
        backend_label=backend_label, log_path=log_path, raw_log_path=raw_data_path,
        plot_path=plot_path, raw=summary, _model_params=params,
    )

    if plot_path is not None or not hide_plot:
        if is_band_structure:
            if is_1d:
                Z_for_plot = Z_exact_bands[:, 0, :]
            else:
                Z_for_plot = Z_exact_bands
        else:
            if is_1d:
                Z_for_plot = Z_exact[:, 0]
            else:
                Z_for_plot = Z_exact
        Z_vqe_plot = Z_vqe[:, 0] if (is_1d and Z_vqe is not None) else Z_vqe
        Z_iqpe_plot = Z_iqpe[:, 0] if (is_1d and Z_iqpe is not None) else Z_iqpe
        plot_simulated(
            x_vals, y_vals, x_label, y_label, Z_for_plot, Z_vqe_plot, Z_iqpe_plot,
            plot_format=plot_format,
            hide_legend=hide_legend,
            output_path=plot_path, hide_plot=hide_plot,
            x_is_momentum=(x_kind == "momentum"),
            y_is_momentum=(y_kind == "momentum"),
        )

    if is_1d:
        Z_exact_out = Z_exact[:, 0]
        Z_vqe_out = Z_vqe[:, 0] if Z_vqe is not None else None
        Z_iqpe_out = Z_iqpe[:, 0] if Z_iqpe is not None else None
        Z_bands_out = Z_exact_bands[:, 0, :] if Z_exact_bands is not None else None
    else:
        Z_exact_out = Z_exact
        Z_vqe_out = Z_vqe
        Z_iqpe_out = Z_iqpe
        Z_bands_out = Z_exact_bands

    return SimulatedResult(
        model_name=model.name,
        lattice=lattice,
        x_param=x_param,
        y_param=y_param,
        x_values=x_vals,
        y_values=y_vals,
        analytic_energies=Z_exact_out,
        vqe_best_energies=Z_vqe_out,
        iqpe_best_energies=Z_iqpe_out,
        plot_format=plot_format,
        band_structure=is_band_structure,
        analytic_bands=Z_bands_out,
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


def _prep_simulated_kwargs(model, lattice, x_param, x_range, y_param, y_range, n_occ, model_params,
                           vqe_iters, vqe_layers, vqe_reps, iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps):
    model = _resolve_model(model)
    vqe_reps = _resolve_method_reps("vqe", vqe_reps, vqe_iters, vqe_layers)
    iqpe_reps = _resolve_method_reps("iqpe", iqpe_reps, iqpe_time, iqpe_trot, iqpe_iters)

    x_param, x_range, y_param, y_range, is_1d = _normalize_sweep_axes(x_param, x_range, y_param, y_range)
    _gate_momentum(model, x_param, y_param)

    if _is_band_structure_axes(model, x_param, y_param):
        if lattice is not None:
            raise ValueError("lattice and momentum-space sweep axes are mutually exclusive; omit lattice for band-structure runs.")
    else:
        lattice = _resolve_lattice(model, lattice)

    for axis in (x_param, y_param):
        if axis is None:
            continue
        if axis != "n_occ" and axis in (model_params or {}):
            raise ValueError(
                f"'{axis}' is the active sweep axis and cannot be set as a fixed value in model_params. "
                f"Override the sweep with x_param/y_param instead."
            )

    params = dict(model_params or {})
    return model, lattice, x_param, x_range, y_param, y_range, is_1d, n_occ, params, vqe_reps, iqpe_reps


def run_simulated_ideal(
    model,
    *,
    lattice=None,
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
    (model, lattice, x_param, x_range, y_param, y_range, is_1d, n_occ, params,
     vqe_reps, iqpe_reps) = _prep_simulated_kwargs(
        model, lattice, x_param, x_range, y_param, y_range, n_occ, model_params,
        vqe_iters, vqe_layers, vqe_reps, iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps,
    )

    return _run_simulated(
        model, "ideal", None,
        lattice=lattice, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ,
        model_params=params, vqe_iters=vqe_iters, vqe_layers=vqe_layers,
        vqe_reps=vqe_reps, iqpe_time=iqpe_time, iqpe_trot=iqpe_trot,
        iqpe_iters=iqpe_iters, iqpe_reps=iqpe_reps,
        log_dir=log_dir, plot_dir=plot_dir,
        hide_plot=hide_plot, hide_legend=hide_legend,
        is_1d=is_1d,
    )


def run_simulated_noisy(
    model,
    *,
    backend=None,
    lattice=None,
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

    (model, lattice, x_param, x_range, y_param, y_range, is_1d, n_occ, params,
     vqe_reps, iqpe_reps) = _prep_simulated_kwargs(
        model, lattice, x_param, x_range, y_param, y_range, n_occ, model_params,
        vqe_iters, vqe_layers, vqe_reps, iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps,
    )

    return _run_simulated(
        model, "noisy", backend,
        lattice=lattice, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ,
        model_params=params, vqe_iters=vqe_iters, vqe_layers=vqe_layers,
        vqe_reps=vqe_reps, iqpe_time=iqpe_time, iqpe_trot=iqpe_trot,
        iqpe_iters=iqpe_iters, iqpe_reps=iqpe_reps,
        log_dir=log_dir, plot_dir=plot_dir,
        hide_plot=hide_plot, hide_legend=hide_legend,
        is_1d=is_1d,
    )
