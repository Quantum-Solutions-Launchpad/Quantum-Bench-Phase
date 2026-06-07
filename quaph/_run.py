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


def _lattice_tag(lattice) -> str:
    if lattice is None:
        return "band-structure"
    return "x".join(str(x) for x in lattice)


def _log_subdir(log_dir, model_name, lattice):
    return os.path.join(log_dir, model_name, _lattice_tag(lattice))


def _plot_subdir(plot_dir, model_name, lattice):
    return os.path.join(plot_dir, model_name, _lattice_tag(lattice))


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


def _run_file_tag(methods, plot_format, x_param, y_param, observable, noisy):
    mstr = "+".join(m.value for m in methods)
    obs = f"-{observable}" if observable and observable != "E" else ""
    noise = "-noisy" if noisy else ""
    if y_param is None:
        return f"run-{mstr}{obs}{noise}-{plot_format}-{x_param}"
    return f"run-{mstr}{obs}{noise}-{plot_format}-{x_param}-vs-{y_param}"


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
class RunResult:
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

    def plot(self, *, hide_plot: bool = False, output_path=None, hide_legend: bool = False):
        return _plot_run_result(
            self, output_path=output_path, hide_plot=hide_plot, hide_legend=hide_legend,
        )


def _squeeze_scalar(arr, is_1d):
    arr = np.asarray(arr, dtype=float)
    return arr[:, 0] if is_1d else arr


def _squeeze_bands(arr, is_1d):
    arr = np.asarray(arr, dtype=float)
    return arr[:, 0, :] if is_1d else arr


def _plot_run_result(rr: RunResult, *, output_path, hide_plot, hide_legend):
    x_label, y_label, x_is_mom, y_is_mom = _result_labels(rr.model_name, rr.x_param, rr.y_param)
    obs_label = _safe_observable_label(rr.model_name, rr.observable)
    methods = [Method.coerce(m) for m in rr.methods]
    plot_format = rr.plot_format
    is_band = rr.band_structure

    if len(methods) == 1:
        m = methods[0]
        if is_band and m == Method.ANALYTIC:
            Z = rr.analytic_bands
        else:
            Z = rr.grids[m.value]
        z_label = obs_label if m == Method.ANALYTIC else f"${get_method_class(m).LABEL}$"
        return plot_analytic(
            rr.x_values, rr.y_values, x_label, y_label if rr.y_param else z_label, Z,
            plot_format=plot_format, output_path=output_path, hide_plot=hide_plot,
            x_is_momentum=x_is_mom, y_is_momentum=y_is_mom, z_label=z_label,
        )

    # Multi-method comparison: pick the reference surface (analytic if selected,
    # else the canonically-first method), render the rest as markers.
    surface_m = Method.ANALYTIC if Method.ANALYTIC in methods else methods[0]
    if is_band and surface_m == Method.ANALYTIC:
        Z_exact = rr.analytic_bands
    else:
        Z_exact = rr.grids[surface_m.value]
    surface_label = get_method_class(surface_m).LABEL

    Z_vqe = rr.grids["vqe"] if (Method.VQE in methods and surface_m != Method.VQE) else None
    Z_iqpe = rr.grids["iqpe"] if (Method.IQPE in methods and surface_m != Method.IQPE) else None
    extra_series = []
    for m in methods:
        if m in (surface_m, Method.VQE, Method.IQPE):
            continue
        style = _METHOD_STYLE.get(m.value, {})
        extra_series.append({
            "label": get_method_class(m).LABEL,
            "values": rr.grids[m.value],
            "color": style.get("color", "#D55E00"),
            "marker": style.get("marker", "D"),
        })

    return plot_simulated(
        rr.x_values, rr.y_values, x_label, y_label, Z_exact, Z_vqe, Z_iqpe,
        plot_format=plot_format, hide_legend=hide_legend,
        output_path=output_path, hide_plot=hide_plot,
        x_is_momentum=x_is_mom, y_is_momentum=y_is_mom,
        surface_label=surface_label, extra_series=extra_series,
    )


def load_result(path: str) -> RunResult:
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
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
    heatmap: bool = False,
    task_index: int | None = None,
    task_count: int = 1,
    prepare_only: bool = False,
    aggregate_only: bool = False,
    no_progress_log: bool = False,
) -> RunResult:
    """Run one or more simulation methods over a parameter sweep.

    ``method`` is a :class:`~quaph.Method` (or list of them). ``method_params`` is
    a dict keyed by method enum/value holding that method's parameters. ``backend``
    selects ideal (``None``) vs. noisy execution for the quantum methods.
    """
    methods = _normalize_methods(method)
    method_objs = _build_method_objects(methods, method_params)
    backend_label = "ideal" if backend is None else type(backend).__name__

    if qubit_operator is not None:
        return _run_operator_methods(
            qubit_operator, methods, method_objs, backend, backend_label,
            extremum=extremum, select=select,
            x_param=x_param, x_range=x_range, y_param=y_param, y_range=y_range,
            heatmap=heatmap, log_dir=log_dir, plot_dir=plot_dir,
            hide_plot=hide_plot, hide_legend=hide_legend,
        )

    return _run_model_methods(
        model, methods, method_objs, backend, backend_label,
        lattice=lattice, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ, model_params=model_params,
        observable=observable, log_dir=log_dir, plot_dir=plot_dir,
        hide_plot=hide_plot, hide_legend=hide_legend, heatmap=heatmap,
        task_index=task_index, task_count=task_count,
        prepare_only=prepare_only, aggregate_only=aggregate_only,
        no_progress_log=no_progress_log,
    )


# --------------------------------------------------------------- model dispatch
def _run_model_methods(
    model, methods, method_objs, backend, backend_label,
    *, lattice, x_param, x_range, y_param, y_range, n_occ, model_params,
    observable, log_dir, plot_dir, hide_plot, hide_legend, heatmap,
    task_index, task_count, prepare_only, aggregate_only, no_progress_log,
):
    from loguru import logger

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
    noisy = backend is not None
    tag = _run_file_tag(methods, plot_format, x_param, y_param, observable, noisy)

    if task_count < 1:
        raise ValueError("task_count must be at least 1")
    if task_index is not None and not 0 <= task_index < task_count:
        raise ValueError("task_index must satisfy 0 <= task_index < task_count")

    use_parallel = any(method_objs[m].WANTS_PARALLEL for m in methods)

    log_subdir = None
    raw_dir = None
    raw_data_path = None
    progress_path = None
    if log_dir is not None:
        log_subdir = _log_subdir(log_dir, model.name, lattice)
        # Raw sidecar + progress journal only matter for expensive parallel runs
        # (benchmarks, resume, sharding); analytic-only runs need just the summary.
        if use_parallel:
            raw_dir = os.path.join(log_subdir, "raw-data")
            os.makedirs(raw_dir, exist_ok=True)
            raw_data_path = os.path.join(raw_dir, f"{tag}.json")
            progress_path = os.path.join(raw_dir, f"{tag}.progress.jsonl")

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
            raw_dir=raw_dir or tmp_dir, tmp_dir=tmp_dir, label=label,
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
            raise ValueError("log_dir is required for aggregate_only")
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

    log_path = None
    if log_subdir is not None:
        os.makedirs(log_subdir, exist_ok=True)
        log_path = os.path.join(log_subdir, f"{tag}.json")
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        plot_subdir = _plot_subdir(plot_dir, model.name, lattice)
        os.makedirs(plot_subdir, exist_ok=True)
        plot_path = os.path.join(plot_subdir, f"{tag}.pdf")

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
        _plot_run_result(result, output_path=plot_path, hide_plot=hide_plot, hide_legend=hide_legend)

    return result


# ------------------------------------------------------------ operator dispatch
def _operator_axis_values(parsed, param, rng):
    tol = 1e-9
    available = sorted({d[param] for d in parsed if param in d})
    if not available:
        raise ValueError(f"No Hamiltonian keys contain a numeric '{param}' token.")
    if rng is None:
        return available
    lo, hi = rng[0], rng[1]
    step = rng[2] if len(rng) > 2 else None
    vals = [v for v in available if lo - tol <= v <= hi + tol]
    if step is not None and step > 0:
        grid = []
        g = lo
        while g <= hi + tol:
            grid.append(g)
            g += step
        snap = max(tol, abs(step) * 1e-6)
        vals = [v for v in vals if any(abs(v - gp) <= snap for gp in grid)]
    if not vals:
        raise ValueError(
            f"No '{param}' token values fall within the requested range for the selected keys "
            f"(available: {available})."
        )
    return vals


def _select_unique_key(keys, parsed, constraints, cell_label):
    tol = 1e-9
    cand = [
        k for k, d in zip(keys, parsed)
        if all(p in d and abs(d[p] - v) <= tol for p, v in constraints)
    ]
    if len(cand) > 1:
        sample = ", ".join(cand[:4]) + (", ..." if len(cand) > 4 else "")
        raise ValueError(
            f"{len(cand)} keys match {cell_label}; the sweep axes don't identify a unique "
            f"Hamiltonian. Narrow the source to one family with select (e.g. select=['1D','grid','pbc']) "
            f"or add the other varying token as a second sweep axis. Matches: {sample}"
        )
    return cand[0] if cand else None


def _filter_keys(keys, select):
    terms = [t.strip() for chunk in (select or []) for t in chunk.split(",") if t.strip()]
    if not terms:
        return keys
    patterns = [re.compile(rf"(?:^|[-_]){re.escape(t)}(?=$|[-_])") for t in terms]
    filtered = [k for k in keys if all(p.search(k) for p in patterns)]
    if not filtered:
        raise ValueError(
            f"No Hamiltonian keys match all select terms {terms}. Terms match whole "
            f"'-'/'_'-delimited segments (e.g. 1D, grid, pbc, or a token like Ly-105)."
        )
    return filtered


def _resolve_operator_axes(keys, x_param, x_range, y_param, y_range):
    if x_param is None and y_param is not None:
        x_param, x_range, y_param, y_range = y_param, y_range, None, None

    if x_param is None:
        x_vals = list(range(len(keys)))
        return "instance", x_vals, "Instance", None, [], None, list(keys), True

    parsed = [parse_key_params(k) for k in keys]
    x_vals = _operator_axis_values(parsed, x_param, x_range)
    x_label = f"${x_param}$"

    if y_param is None:
        grid = [
            _select_unique_key(keys, parsed, [(x_param, xv)], f"{x_param}={xv}")
            for xv in x_vals
        ]
        return x_param, x_vals, x_label, None, [], None, grid, True

    y_vals = _operator_axis_values(parsed, y_param, y_range)
    grid = [
        [
            _select_unique_key(
                keys, parsed, [(x_param, xv), (y_param, yv)],
                f"{x_param}={xv}, {y_param}={yv}",
            )
            for yv in y_vals
        ]
        for xv in x_vals
    ]
    return x_param, x_vals, x_label, y_param, y_vals, f"${y_param}$", grid, False


def _run_operator_methods(
    qubit_operator, methods, method_objs, backend, backend_label,
    *, extremum, select, x_param, x_range, y_param, y_range,
    heatmap, log_dir, plot_dir, hide_plot, hide_legend,
):
    from loguru import logger

    if extremum not in ("min", "max"):
        raise ValueError(f"extremum must be 'min' or 'max'; got {extremum!r}.")
    for m in methods:
        if not method_objs[m].SUPPORTS_OPERATOR:
            raise ValueError(
                f"Method '{m.value}' does not support the --qubit-operator path."
            )
    if heatmap and len(methods) != 1:
        raise ValueError("heatmap=True requires exactly one simulation method.")

    path = qubit_operator
    keys = list_hamlib_keys(path)
    if not keys:
        raise ValueError(f"No Hamiltonian datasets found in '{path}'.")
    keys = _filter_keys(keys, select)

    (x_param, x_vals, x_label, y_param, y_vals, y_label, key_grid,
     is_1d) = _resolve_operator_axes(keys, x_param, x_range, y_param, y_range)
    if heatmap and is_1d:
        raise ValueError("heatmap requires both x and y sweep axes; provide y_param/y_range.")
    nx = len(x_vals)
    ny = 1 if is_1d else len(y_vals)
    model_name = os.path.splitext(os.path.basename(path))[0]
    noisy = backend is not None
    plot_format = "2d" if is_1d else ("heatmap" if heatmap else "3d")

    def cell_key(ix, iy):
        return key_grid[ix] if is_1d else key_grid[ix][iy]

    def cell_label(ix, iy):
        if is_1d:
            return f"{x_param}={x_vals[ix]}"
        return f"{x_param}={x_vals[ix]}, {y_param}={y_vals[iy]}"

    raw_cells = {m.value: {str(ix): {} for ix in range(nx)} for m in methods}

    def run_job(method_value, ix, iy):
        key = cell_key(ix, iy)
        if key is None:
            return (method_value, ix, iy), None
        op = load_hamlib_operator(path, key)
        m_obj = method_objs[Method.coerce(method_value)]
        cell = m_obj.compute_operator_cell(op, extremum=extremum, backend=backend, label=cell_label(ix, iy))
        return (method_value, ix, iy), cell

    def init_worker_logging():
        from quaph._core import setup_logging as _sl
        _sl()

    all_tags = [(m.value, ix, iy) for m in methods for ix in range(nx) for iy in range(ny)]
    jobs = (delayed(run_job)(*t) for t in all_tags)
    for (mv, ix, iy), cell in Parallel(
        n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging,
    )(jobs):
        if cell is not None:
            raw_cells[mv][str(ix)][str(iy)] = cell

    grids_full = {}
    for m in methods:
        m_obj = method_objs[m]
        arr = np.full((nx, ny), np.nan)
        for ix in range(nx):
            for iy in range(ny):
                cell = raw_cells[m.value].get(str(ix), {}).get(str(iy))
                if cell is None:
                    continue
                arr[ix, iy] = m_obj.reduce(cell, extremum=extremum)
                logger.info(f"{m_obj.LABEL} ({cell_label(ix, iy)}) = {arr[ix, iy]}")
        grids_full[m.value] = arr

    def scalar_block(arr):
        if is_1d:
            return {ix: float(arr[ix, 0]) for ix in range(nx)}
        return {ix: {iy: float(arr[ix, iy]) for iy in range(ny)} for ix in range(nx)}

    summary = {
        "type": "run",
        "methods": [m.value for m in methods],
        "backend": backend_label,
        "plot_format": plot_format,
        "band_structure": False,
        "observable": "E",
        "extremum": extremum,
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "parameters": {
            "model": model_name,
            "lattice": None,
            "qubit_operator": path,
            "keys": key_grid,
            "extremum": extremum,
            "model_params": {},
            "method_params": {m.value: method_objs[m].parameter_summary() for m in methods},
        },
        "result": {m.value: scalar_block(grids_full[m.value]) for m in methods},
    }

    tag = _run_file_tag(methods, plot_format, x_param, y_param, "E", noisy)
    log_path = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{tag}.json")
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{tag}.pdf")

    grids_out = {m.value: _squeeze_scalar(grids_full[m.value], is_1d) for m in methods}
    result = RunResult(
        model_name=model_name, lattice=None, x_param=x_param, y_param=y_param,
        x_values=x_vals, y_values=y_vals if not is_1d else [],
        methods=[m.value for m in methods], grids=grids_out,
        band_structure=False, plot_format=plot_format, observable="E",
        extremum=extremum, backend_label=backend_label,
        log_path=log_path, plot_path=plot_path, raw=summary, _model_params={},
    )

    if plot_path is not None or not hide_plot:
        # Operator labels come straight from the key tokens, not a registered model.
        x_lab = x_label
        y_lab = y_label or "$E$"
        _plot_operator_result(result, x_lab, y_lab, methods,
                              output_path=plot_path, hide_plot=hide_plot, hide_legend=hide_legend)

    return result


def _plot_operator_result(rr, x_label, y_label, methods, *, output_path, hide_plot, hide_legend):
    if len(methods) == 1:
        m = methods[0]
        z_label = "$E$" if m == Method.ANALYTIC else f"${get_method_class(m).LABEL}$"
        return plot_analytic(
            rr.x_values, rr.y_values, x_label, y_label if rr.y_param else z_label,
            rr.grids[m.value], plot_format=rr.plot_format,
            output_path=output_path, hide_plot=hide_plot,
            x_is_momentum=False, y_is_momentum=False, z_label=z_label,
        )
    surface_m = Method.ANALYTIC if Method.ANALYTIC in methods else methods[0]
    Z_exact = rr.grids[surface_m.value]
    Z_vqe = rr.grids["vqe"] if (Method.VQE in methods and surface_m != Method.VQE) else None
    Z_iqpe = rr.grids["iqpe"] if (Method.IQPE in methods and surface_m != Method.IQPE) else None
    extra_series = []
    for m in methods:
        if m in (surface_m, Method.VQE, Method.IQPE):
            continue
        style = _METHOD_STYLE.get(m.value, {})
        extra_series.append({
            "label": get_method_class(m).LABEL, "values": rr.grids[m.value],
            "color": style.get("color", "#D55E00"), "marker": style.get("marker", "D"),
        })
    return plot_simulated(
        rr.x_values, rr.y_values, x_label, y_label, Z_exact, Z_vqe, Z_iqpe,
        plot_format=rr.plot_format, hide_legend=hide_legend,
        output_path=output_path, hide_plot=hide_plot,
        surface_label=get_method_class(surface_m).LABEL, extra_series=extra_series,
    )
