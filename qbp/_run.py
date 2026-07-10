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

from qbp._model import Model, ModelCapabilityError, matrix_to_fermionic_op
from qbp._backend import resolve_backend, backend_label as _backend_label, is_real_backend
from qbp._core import resolve_sweep
from qbp._hamlib import (
    list_hamlib_keys, load_hamlib_operator, parse_key_params,
    collect_keys_multi,
)
from qbp._geometry import (
    apply_geometry_to_hamiltonian,
    geometry_projection,
    normalize_geometry,
    project_fermionic_op,
)
from qbp._plotting import plot_analytic, plot_simulated
from qbp._profiles import apply_profiles_to_hamiltonian, profile_metadata
from qbp._boundary import _normalize_boundary, _with_boundary, _resolve_boundary
from qbp._diff import _diff_3d, _diff_heatmap, _diff_bar_2d
from qbp._registry import get_model as _get_model
from qbp._method import (
    Method, METHOD_ORDER, CellContext, build_method, get_method_class,
)
from qbp._investigation import build_investigation


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
    from qbp._registry import get_model
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
    from qbp._registry import get_model
    try:
        return f"${get_model(model_name).get_observable(observable).display_name}$"
    except Exception:
        return "$E$"


# Real-space diagnostics select a whole plot type (not a sweep scalar), so they
# live on a dedicated `plot` axis rather than on `observable`.
_PLOT_KIND_ALIASES = {
    "energy": "energy",
    "sweep": "energy",
    "real_space_density": "real_space_density",
    "real_space": "real_space_density",
    "density": "real_space_density",
    "state_density": "real_space_density",
    "edge_spectrum": "edge_spectrum",
    "edge": "edge_spectrum",
}


def _normalize_plot_kind(plot: str | None) -> str:
    if plot is None:
        return "energy"
    key = str(plot).strip().lower().replace("-", "_").replace(" ", "_")
    kind = _PLOT_KIND_ALIASES.get(key)
    if kind is None:
        raise ValueError(
            f"unsupported plot kind {plot!r}; expected 'energy', "
            f"'real_space_density', or 'edge_spectrum'."
        )
    return kind


_SPECTRAL_AXES = ("eigenstate",)


def _diagnostic_kind_from_axes(model, x_param, y_param) -> str:
    axes = [a for a in (x_param, y_param) if a is not None]
    spatial_names = set(model.lattice_shape)
    spatial = [a for a in axes if a in spatial_names]
    spectral = [a for a in axes if a in _SPECTRAL_AXES]
    if not spatial and not spectral:
        return "energy"
    if spectral:
        if len(axes) != 1:
            raise ValueError(
                "'eigenstate' is a standalone edge-spectrum axis and cannot be paired "
                "with another sweep axis."
            )
        return "edge_spectrum"
    if len(axes) != len(spatial):
        raise ValueError(
            f"real-space axes {sorted(spatial_names)} cannot be combined with non-spatial "
            f"sweep axes."
        )
    return "real_space_density"


def _serialize_model_params(params: dict) -> dict:
    serial = {}
    for k, v in params.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            serial[k] = float(v)
        else:
            serial[k] = v
    return serial


def _build_modified_H(model, lattice, model_params, projection, potential_kwargs, investigation):
    H = apply_geometry_to_hamiltonian(model._build_H_matrix(lattice, **model_params), projection)
    H = apply_profiles_to_hamiltonian(H, model, projection, model_params, **potential_kwargs)
    if investigation is not None:
        H = investigation.apply(H, model, projection, model_params)
    return H


def _modified_fermionic_fn(model, projection, potential_kwargs, investigation):
    def _build(lattice, **params):
        H = _build_modified_H(model, lattice, params, projection, potential_kwargs, investigation)
        op = matrix_to_fermionic_op(H)
        if model._interaction_hamiltonian_fn is not None:
            int_op = model._interaction_hamiltonian_fn(lattice, **params)
            op = op + project_fermionic_op(int_op, projection)
        return op
    return _build


def _analytic_projected_cell(model, lattice, n_occ, model_params, observable, projection,
                             potential_kwargs, investigation):
    from loguru import logger
    from qbp._core import _fmt_params

    H = _build_modified_H(model, lattice, model_params, projection, potential_kwargs, investigation)
    eigvals, eigvecs = np.linalg.eigh(H)
    obs = model.get_observable(observable)
    model._analytic_fermionic_fn = _modified_fermionic_fn(
        model, projection, potential_kwargs, investigation
    )
    model._analytic_projection_sig = (projection.geometry, int(projection.orbital_mask.sum()))
    try:
        result = float(obs.analytic(model, lattice, H, eigvals, eigvecs, n_occ, model_params))
    finally:
        model._analytic_fermionic_fn = None
        model._analytic_projection_sig = None
    logger.info(
        f"Analytic [{observable}] ({_fmt_params(lattice, n_occ, model_params)}, "
        f"geometry={projection.geometry}, active_sites={int(projection.site_mask.sum())}) = {result}"
    )
    return result


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
    plot_kind: str = "energy"
    diagnostic: object = field(default=None, repr=False)
    raw: dict = field(default_factory=dict, repr=False)
    _model_params: dict = field(default_factory=dict, repr=False)
    _replot: object = field(default=None, repr=False)

    def plot(self, *, hide_plot: bool = False, output_path=None,
            hide_legend: bool = False, diff: bool = False,
            diff_format: str = "3d"):
        if self._replot is not None:
            self.diagnostic = self._replot(output_path=output_path, hide_plot=hide_plot)
            return self.diagnostic.figure
        result = _plot_run_result(
            self, output_path=output_path, hide_plot=hide_plot, hide_legend=hide_legend,
        )
        if diff:
            x_label, y_label, x_is_mom, y_is_mom = _result_labels(
                self.model_name, self.x_param, self.y_param
            )
            _plot_diffs(
                self, x_label=x_label, y_label=y_label,
                x_is_momentum=x_is_mom, y_is_momentum=y_is_mom,
                plot_format=diff_format, output_path=output_path, hide_plot=hide_plot,
            )
        return result

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


def _diff_output_path(base, tag):
    """Insert a ``-diff-<tag>`` suffix before the extension of an output path."""
    if base is None:
        return None
    if "." in os.path.basename(base):
        stem, ext = base.rsplit(".", 1)
        return f"{stem}-diff-{tag}.{ext}"
    return f"{base}-diff-{tag}"


def _plot_diffs(rr: RunResult, *, x_label, y_label, x_is_momentum, y_is_momentum,
                plot_format, output_path, hide_plot):
    """Render a difference plot for every pair of methods in ``rr``.

    Each plot shows ``E_b - E_a`` where ``a`` precedes ``b`` in METHOD_ORDER, so
    quantum methods are differenced against the analytic reference by convention.
    Produced in addition to (not in place of) the normal plots.
    """
    from itertools import combinations

    methods = [Method.coerce(m) for m in rr.methods]
    figs = []
    for a, b in combinations(methods, 2):
        Z_err = np.asarray(rr.grids[b.value], dtype=float) - np.asarray(rr.grids[a.value], dtype=float)
        if Z_err.ndim == 1:
            Z_err = Z_err[:, None]
        la = get_method_class(a).LABEL
        lb = get_method_class(b).LABEL
        z_label = rf"$E_{{\mathrm{{{lb}}}}} - E_{{\mathrm{{{la}}}}}$"
        shared = dict(
            x_label=x_label, y_label=y_label, z_label=z_label,
            x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
            output_path=_diff_output_path(output_path, f"{b.value}-{a.value}"),
            hide_plot=hide_plot,
        )
        if plot_format == "heatmap":
            fig = _diff_heatmap(rr.x_values, rr.y_values, Z_err, **shared)
        elif plot_format == "bar_2d":
            fig = _diff_bar_2d(rr.x_values, rr.y_values, Z_err, **shared)
        else:
            fig = _diff_3d(rr.x_values, rr.y_values, Z_err,
                           hover_label=f"{lb} − {la}", **shared)
        figs.append(fig)
    return figs


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


# ------------------------------------------------- nested op-dict flattening
def _flatten_op_dict(node, _prefix="") -> dict:
    """Flatten a nested parameter dict into a flat {key_string: SparsePauliOp} dict.

    The input alternates between string keys (parameter/prefix names) and
    non-string keys (parameter values), terminating at SparsePauliOp leaves:

        {"stem": {"h": {0.5: {"n": {4: op, 6: op}}}}}
        → {"stem_h-0.5_n-4": op, "stem_h-0.5_n-6": op}

    A flat dict (values already SparsePauliOp) passes through unchanged.
    """
    from qiskit.quantum_info import SparsePauliOp as _SPO

    if isinstance(node, _SPO):
        return {_prefix: node}

    result: dict = {}
    first_key = next(iter(node))

    if isinstance(first_key, str):
        for key, child in node.items():
            new_prefix = f"{_prefix}_{key}" if _prefix else key
            result.update(_flatten_op_dict(child, new_prefix))
    else:
        for key, child in node.items():
            val_str = str(int(key)) if isinstance(key, int) else f"{key:g}"
            result.update(_flatten_op_dict(child, f"{_prefix}-{val_str}"))

    return result


# ----------------------------------------------------------- public entry point
def run(
    model=None,
    *,
    method,
    method_params: dict | None = None,
    lattice=None,
    boundary: str | None = None,
    boundary_params: dict | None = None,
    investigation=None,
    investigation_params: dict | None = None,
    x_param: str | None = None,
    x_range=None,
    y_param: str | None = None,
    y_range=None,
    n_occ: int | None = None,
    model_params: dict | None = None,
    observable: str = "E",
    backend=None,
    qubit_operator: str | list[str] | dict | None = None,
    extremum: str = "min",
    select=None,
    log_path=None,
    plot_path=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
    heatmap: bool = False,
    diff: bool = False,
    diff_format: str = "3d",
    task_index: int | None = None,
    task_count: int = 1,
    prepare_only: bool = False,
    aggregate_only: bool = False,
    no_progress_log: bool = False,
) -> RunResult:
    """Run one or more simulation methods over a parameter sweep.

    ``method`` is a :class:`~qbp.Method` (or list of them). ``method_params`` is
    a dict keyed by method enum/value holding that method's parameters. ``backend``
    selects how the quantum methods run: ``None`` for ideal simulation, a fake
    backend (object or name, e.g. ``"FakeSherbrooke"``) for local noisy
    simulation, or a real IBM device for hardware execution -- given as a device
    name (e.g. ``"ibm_brisbane"``), ``"least_busy"``, or an ``IBMBackend`` object,
    and requiring configured Qiskit Runtime credentials.

    ``log_path`` / ``plot_path`` are the exact JSON / PDF files to write (both the
    containing directory and the file name are chosen by the caller); parent
    directories are created as needed. Either may be ``None`` to skip that output.

    ``boundary`` selects ``'periodic'`` or ``'open'``; ``boundary_params`` is the
    open-boundary parameter dict (``geometry``/``radius``/``center`` and the
    ``potential_*`` soft-dot knobs), paired with ``boundary`` the same way
    ``model_params`` is paired with ``model``. Periodic boundaries take no
    parameters.

    ``investigation`` selects a model-specific physics study (an
    :class:`~qbp._investigation.Investigation` instance or registered name, e.g.
    ``"semenoff_mass"``); ``investigation_params`` supplies its parameters when
    selecting by name. It is paired with ``investigation`` the same way
    ``method_params`` is paired with ``method``, and applies to real-space
    analytic runs and diagnostics only.

    The sweep axes select the kind of figure, just as momentum axes (``kx``/``ky``)
    select a band-structure run. The model's real-space lattice axes (``Lx``/``Ly``)
    render a single-particle real-space eigenstate-density map, and ``eigenstate``
    renders an edge-participation spectrum. Both are exact-diagonalization
    diagnostics of one Hamiltonian (``method=Method.ANALYTIC``); the boundary
    condition, open-boundary geometry/potential and the selected investigation all
    apply.
    """
    methods = _normalize_methods(method)
    boundary, bparams = _resolve_boundary(boundary, boundary_params)
    investigation = build_investigation(investigation, investigation_params)
    geometry = bparams["geometry"]
    radius = bparams["radius"]
    center = bparams["center"]
    potential_profile = bparams["potential_profile"]
    potential_radius = bparams["potential_radius"]
    potential_v0 = bparams["potential_v0"]
    potential_xi = bparams["potential_xi"]

    if model is not None and qubit_operator is None:
        diagnostic_model = _resolve_model(model)
        diagnostic_kind = _diagnostic_kind_from_axes(diagnostic_model, x_param, y_param)
        if diagnostic_kind != "energy":
            return _run_diagnostic(
                diagnostic_kind, diagnostic_model, methods,
                lattice=lattice, boundary=boundary, geometry=geometry,
                radius=radius, center=center, n_occ=n_occ, model_params=model_params,
                potential_profile=potential_profile, potential_radius=potential_radius,
                potential_v0=potential_v0, potential_xi=potential_xi,
                investigation=investigation,
                plot_path=plot_path, hide_plot=hide_plot,
            )

    method_objs = _build_method_objects(methods, method_params)
    backend = resolve_backend(backend)
    backend_label = _backend_label(backend)

    if qubit_operator is not None:
        if isinstance(qubit_operator, str):
            qubit_operator = [qubit_operator]
        elif isinstance(qubit_operator, dict):
            qubit_operator = _flatten_op_dict(qubit_operator)
        else:
            qubit_operator = list(qubit_operator)
        return _run_operator_methods(
            qubit_operator, methods, method_objs, backend, backend_label,
            extremum=extremum, select=select, observable=observable,
            x_param=x_param, x_range=x_range, y_param=y_param, y_range=y_range,
            heatmap=heatmap, log_path=log_path, plot_path=plot_path,
            hide_plot=hide_plot, hide_legend=hide_legend,
            diff=diff, diff_format=diff_format,
        )

    return _run_model_methods(
        model, methods, method_objs, backend, backend_label,
        lattice=lattice, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ, model_params=model_params,
        boundary=boundary, geometry=geometry, radius=radius, center=center,
        potential_profile=potential_profile, potential_radius=potential_radius,
        potential_v0=potential_v0, potential_xi=potential_xi,
        investigation=investigation,
        observable=observable, log_path=log_path, plot_path=plot_path,
        hide_plot=hide_plot, hide_legend=hide_legend, heatmap=heatmap,
        diff=diff, diff_format=diff_format,
        task_index=task_index, task_count=task_count,
        prepare_only=prepare_only, aggregate_only=aggregate_only,
        no_progress_log=no_progress_log,
    )


# --------------------------------------------------------- real-space diagnostics
def _run_diagnostic(
    plot_kind, model, methods, *,
    lattice, boundary, geometry, radius, center, n_occ, model_params,
    potential_profile, potential_radius, potential_v0, potential_xi,
    investigation,
    plot_path, hide_plot,
):
    if methods != [Method.ANALYTIC]:
        raise ValueError(
            f"the '{plot_kind}' diagnostic is a single-particle exact-diagonalization "
            f"plot; use method=Method.ANALYTIC."
        )
    lat = _resolve_lattice(model, lattice)
    common = dict(
        model=model, lattice=lat, model_params=model_params, boundary=boundary,
        geometry=geometry, radius=radius, center=center,
        potential_profile=potential_profile, potential_radius=potential_radius,
        potential_v0=potential_v0, potential_xi=potential_xi,
        investigation=investigation,
    )
    if plot_kind == "real_space_density":
        from qbp._real_space import plot_real_space_state_density as _diag_fn
        common["n_occ"] = n_occ
        shape = tuple(model.lattice_shape)
        x_param = shape[0]
        y_param = shape[1] if len(shape) > 1 else None
    else:
        from qbp._edge import plot_edge_spectrum as _diag_fn
        x_param, y_param = "eigenstate", None

    def _replot(*, output_path, hide_plot):
        return _diag_fn(output_path=output_path, hide_plot=hide_plot, **common)

    diagnostic = _replot(output_path=plot_path, hide_plot=hide_plot)
    return RunResult(
        model_name=model.name, lattice=lat, x_param=x_param, y_param=y_param,
        x_values=[], y_values=[], methods=["analytic"], grids={},
        plot_format=plot_kind, observable="E", backend_label="ideal",
        plot_path=plot_path, plot_kind=plot_kind, diagnostic=diagnostic,
        _model_params=dict(model_params or {}), _replot=_replot,
    )


# --------------------------------------------------------------- model dispatch
def _run_model_methods(
    model, methods, method_objs, backend, backend_label,
    *, lattice, x_param, x_range, y_param, y_range, n_occ, model_params,
    boundary, geometry, radius, center,
    potential_profile, potential_radius, potential_v0, potential_xi,
    investigation,
    observable, log_path, plot_path, hide_plot, hide_legend, heatmap,
    diff, diff_format,
    task_index, task_count, prepare_only, aggregate_only, no_progress_log,
):
    from loguru import logger

    model = _resolve_model(model)
    _ = model._build_H_matrix
    _ = model.get_observable(observable)
    if investigation is not None:
        investigation.check_model(model)

    x_param, x_range, y_param, y_range, is_1d = _normalize_sweep_axes(
        x_param, x_range, y_param, y_range
    )
    if heatmap and is_1d:
        raise ValueError("heatmap=True requires both x and y sweep axes; provide y_param/y_range.")
    if heatmap and len(methods) != 1:
        raise ValueError("heatmap=True requires exactly one simulation method.")
    _gate_momentum(model, x_param, y_param)

    params = _with_boundary(model_params, boundary)
    geometry_mode = normalize_geometry(geometry)
    potential_kwargs = dict(
        potential_profile=potential_profile,
        potential_radius=potential_radius,
        potential_v0=potential_v0,
        potential_xi=potential_xi,
        center=center,
    )
    profile_info = profile_metadata(
        potential_profile=potential_profile,
        potential_radius=potential_radius,
        potential_v0=potential_v0,
        potential_xi=potential_xi,
    )
    has_profiles = (profile_info["potential_profile"] != "none") or (investigation is not None)

    is_band_structure_run = _is_band_structure_axes(model, x_param, y_param)
    if is_band_structure_run:
        if lattice is not None:
            raise ValueError("lattice and momentum-space sweep axes are mutually exclusive; omit lattice for band-structure runs.")
        if geometry_mode != "rectangle":
            raise ValueError("disk geometry is only supported for real-space lattice runs, not momentum-space band-structure runs.")
        if _normalize_boundary(params.get("boundary")) != "periodic":
            raise ValueError("open boundary is only supported for real-space lattice runs, not momentum-space band-structure runs.")
        if has_profiles:
            raise ValueError("open-boundary potential profiles and investigations are only supported for real-space lattice runs.")
        for m in methods:
            if not method_objs[m].SUPPORTS_BAND_STRUCTURE:
                raise ValueError(
                    f"Method '{m.value}' does not support band-structure (momentum-space) runs."
                )
        projection = None
    else:
        lattice = _resolve_lattice(model, lattice)
        projection = geometry_projection(
            model,
            lattice,
            geometry=geometry_mode,
            radius=radius,
            center=center,
        )

    for axis in (x_param, y_param):
        if axis is None:
            continue
        if axis != "n_occ" and axis in (model_params or {}):
            raise ValueError(
                f"'{axis}' is the active sweep axis and cannot be set as a fixed value in model_params. "
                f"Override the sweep with x_param/y_param instead."
            )

    if Method.IQPE in methods and observable != "E":
        from qbp._iqpe import iqpe_supports_observable
        if not iqpe_supports_observable(model, observable):
            raise ValueError(
                f"IQPE cannot measure observable '{observable}'; only 'E' and energy "
                f"composites (e.g. charge_gap) are supported. Drop IQPE or use VQE."
            )

    spin = model.spin
    if projection is not None:
        n_sites = int(projection.site_mask.sum())
        n_orbitals = n_sites * spin
    elif lattice is not None:
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

    use_parallel = any(method_objs[m].WANTS_PARALLEL for m in methods) and not is_real_backend(backend)

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
        projected = projection is not None and (projection.geometry != "rectangle" or has_profiles)
        if projected and m_obj.METHOD == Method.ANALYTIC:
            value = _analytic_projected_cell(
                model,
                lattice,
                n_occ_val,
                cp,
                observable,
                projection,
                potential_kwargs,
                investigation,
            )
            return {"value": value}
        if projected:
            ctx.fermionic_hamiltonian_fn = _modified_fermionic_fn(
                model, projection, potential_kwargs, investigation
            )
        else:
            ctx.fermionic_hamiltonian_fn = model.fermionic_hamiltonian
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
                "geometry": projection.geometry if projection is not None else "rectangle",
                "radius": radius,
                "center": list(center) if center is not None else None,
                **profile_info,
                "investigation": investigation.metadata() if investigation is not None else None,
                "model_params": _serialize_model_params(params),
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
        from qbp._core import setup_logging as _sl
        _sl()

    def jobs_per_shard():
        value = os.environ.get("QBP_JOBS_PER_SHARD")
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            return max(1, os.cpu_count() or 1)

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

        with tempfile.TemporaryDirectory(prefix="qbp-run-") as tmp_dir:
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
            "geometry": projection.geometry if projection is not None else "rectangle",
            "radius": radius,
            "center": list(center) if center is not None else None,
            **profile_info,
            "investigation": investigation.metadata() if investigation is not None else None,
            "model_params": _serialize_model_params(params),
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
        _plot_run_result(result, output_path=plot_path, hide_plot=hide_plot, hide_legend=hide_legend)
        if diff:
            x_label, y_label, x_is_mom, y_is_mom = _result_labels(result.model_name, x_param, y_param)
            _plot_diffs(
                result, x_label=x_label, y_label=y_label,
                x_is_momentum=x_is_mom, y_is_momentum=y_is_mom,
                plot_format=diff_format, output_path=plot_path, hide_plot=hide_plot,
            )

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
    *, extremum, select, observable, x_param, x_range, y_param, y_range,
    heatmap, log_path, plot_path, hide_plot, hide_legend,
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

    if isinstance(qubit_operator, dict):
        all_display_keys = list(qubit_operator.keys())
        if not all_display_keys:
            raise ValueError("qubit_operator dict is empty.")
        display_keys = _filter_keys(all_display_keys, select)
        model_name = os.path.commonprefix(display_keys).rstrip("-_") or "custom"

        def _load_op(display_key):
            return qubit_operator[display_key]

        source_repr = "dict"
    else:
        paths = qubit_operator
        all_display_keys, key_to_source = collect_keys_multi(paths)
        if not all_display_keys:
            raise ValueError("No Hamiltonian datasets found in the provided source(s).")
        display_keys = _filter_keys(all_display_keys, select)
        if len(paths) == 1:
            model_name = os.path.splitext(os.path.basename(paths[0]))[0]
        else:
            model_name = os.path.commonprefix(
                [os.path.splitext(os.path.basename(p))[0] for p in paths]
            ).rstrip("-_") or os.path.splitext(os.path.basename(paths[0]))[0]

        def _load_op(display_key):
            file_path, actual_key = key_to_source[display_key]
            return load_hamlib_operator(file_path, actual_key)

        source_repr = paths if len(paths) > 1 else paths[0]

    (x_param, x_vals, x_label, y_param, y_vals, y_label, key_grid,
     is_1d) = _resolve_operator_axes(display_keys, x_param, x_range, y_param, y_range)
    if heatmap and is_1d:
        raise ValueError("heatmap requires both x and y sweep axes; provide y_param/y_range.")
    nx = len(x_vals)
    ny = 1 if is_1d else len(y_vals)
    plot_format = "2d" if is_1d else ("heatmap" if heatmap else "3d")

    def cell_key(ix, iy):
        return key_grid[ix] if is_1d else key_grid[ix][iy]

    def cell_label(ix, iy):
        if is_1d:
            return f"{x_param}={x_vals[ix]}"
        return f"{x_param}={x_vals[ix]}, {y_param}={y_vals[iy]}"

    raw_cells = {m.value: {str(ix): {} for ix in range(nx)} for m in methods}

    def run_job(method_value, ix, iy):
        display_key = cell_key(ix, iy)
        if display_key is None:
            return (method_value, ix, iy), None
        op = _load_op(display_key)
        m_obj = method_objs[Method.coerce(method_value)]
        cell = m_obj.compute_operator_cell(op, extremum=extremum, backend=backend, label=cell_label(ix, iy), observable=observable)
        return (method_value, ix, iy), cell

    def init_worker_logging():
        from qbp._core import setup_logging as _sl
        _sl()

    all_tags = [(m.value, ix, iy) for m in methods for ix in range(nx) for iy in range(ny)]
    jobs = (delayed(run_job)(*t) for t in all_tags)
    for (mv, ix, iy), cell in Parallel(
        n_jobs=1 if is_real_backend(backend) else -1,
        return_as="generator_unordered", initializer=init_worker_logging,
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
        "observable": observable,
        "extremum": extremum,
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "parameters": {
            "model": model_name,
            "lattice": None,
            "qubit_operator": source_repr,
            "keys": key_grid,
            "extremum": extremum,
            "model_params": {},
            "method_params": {m.value: method_objs[m].parameter_summary() for m in methods},
        },
        "result": {m.value: scalar_block(grids_full[m.value]) for m in methods},
    }

    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=4)

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
        if diff:
            _plot_diffs(
                result, x_label=x_lab, y_label=y_lab,
                x_is_momentum=False, y_is_momentum=False,
                plot_format=diff_format, output_path=plot_path, hide_plot=hide_plot,
            )

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
