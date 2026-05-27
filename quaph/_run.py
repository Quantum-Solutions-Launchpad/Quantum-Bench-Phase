from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from quaph._model import Model, ModelCapabilityError
from quaph._core import (
    resolve_sweep,
    analytic, vqe_fermionic, iqpe_fermionic, vqe_observable, iqpe_observable,
    vqe_other_benchmarks, iqpe_other_benchmarks,
    analytic_bands, vqe_bloch, iqpe_bloch,
    vqe_bloch_other_benchmarks, iqpe_bloch_other_benchmarks,
    analytic_operator, vqe_operator, iqpe_operator,
)
from quaph._hamlib import (
    parse_operator_spec, list_hamlib_keys, load_hamlib_operator, parse_key_param,
)
from quaph._plotting import plot_analytic, plot_simulated
from quaph._registry import get_model as _get_model


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


def _observable_label(model, observable: str) -> str:
    obs = model.get_observable(observable)
    return f"${obs.display_name}$"


def _file_tag(run_type: str, plot_format: str, x_param: str, y_param: str | None,
              observable: str | None = None) -> str:
    obs = f"-{observable}" if observable and observable != "E" else ""
    if y_param is None:
        return f"{run_type}{obs}-{plot_format}-{x_param}"
    return f"{run_type}{obs}-{plot_format}-{x_param}-vs-{y_param}"


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


@dataclass
class AnalyticResult:
    model_name: str
    lattice: tuple[int, ...]
    x_param: str
    y_param: str | None
    x_values: list
    y_values: list
    energies: np.ndarray
    plot_format: str = "3d"
    band_structure: bool = False
    log_path: str | None = None
    plot_path: str | None = None
    _model_params: dict = field(default_factory=dict, repr=False)

    def plot(self, *, hide_plot: bool = False, output_path=None):
        x_label, y_label, x_is_momentum, y_is_momentum = _result_labels(
            self.model_name, self.x_param, self.y_param
        )
        return plot_analytic(
            self.x_values, self.y_values, x_label, y_label, self.energies,
            plot_format=self.plot_format,
            output_path=output_path, hide_plot=hide_plot,
            x_is_momentum=x_is_momentum, y_is_momentum=y_is_momentum,
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
        x_label, y_label, x_is_momentum, y_is_momentum = _result_labels(
            self.model_name, self.x_param, self.y_param
        )
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

    result_type = data.get("type", "")
    x_vals = data["x_values"]
    y_vals = data.get("y_values", [])
    is_1d = y_vals is None or len(y_vals) == 0
    nx = len(x_vals)
    ny = len(y_vals) if not is_1d else 0
    band_structure = bool(data.get("band_structure", False))
    plot_format = data.get("plot_format", "2d" if is_1d else "3d")

    def _read_grid(block):
        if is_1d:
            return np.array([block[str(ix)] for ix in range(nx)])
        return np.array([[block[str(ix)][str(iy)] for iy in range(ny)] for ix in range(nx)])

    if result_type == "analytic":
        energies = _read_grid(data["result"]["analytic"])
        return AnalyticResult(
            model_name=data["parameters"]["model"],
            lattice=_opt_lattice(data["parameters"].get("lattice")),
            x_param=data["x_param"],
            y_param=data.get("y_param"),
            x_values=x_vals,
            y_values=y_vals if not is_1d else [],
            energies=energies,
            plot_format=plot_format,
            band_structure=band_structure,
            log_path=path,
            _model_params=data["parameters"].get("model_params", {}),
        )
    elif result_type in ("simulated-ideal", "simulated-noisy"):
        analytic_arr = _read_grid(data["result"]["analytic"])
        if band_structure:
            analytic_bands_arr = analytic_arr
            Z_exact = analytic_arr[..., 0]
        else:
            analytic_bands_arr = None
            Z_exact = analytic_arr
        Z_vqe = _read_grid(data["result"]["vqe"]) if "vqe" in data["result"] else None
        Z_iqpe = _read_grid(data["result"]["iqpe"]) if "iqpe" in data["result"] else None
        return SimulatedResult(
            model_name=data["parameters"]["model"],
            lattice=_opt_lattice(data["parameters"].get("lattice")),
            x_param=data["x_param"],
            y_param=data.get("y_param"),
            x_values=x_vals,
            y_values=y_vals if not is_1d else [],
            analytic_energies=Z_exact,
            vqe_best_energies=Z_vqe,
            iqpe_best_energies=Z_iqpe,
            plot_format=plot_format,
            band_structure=band_structure,
            analytic_bands=analytic_bands_arr,
            raw=data,
            summary_log_path=path,
            _model_params=data["parameters"].get("model_params", {}),
        )
    else:
        raise ValueError(
            f"Unrecognized log file type '{result_type}' in {path}. "
            "Expected 'analytic', 'simulated-ideal', or 'simulated-noisy'."
        )


def _operator_subdir(base, model_name):
    return os.path.join(base, model_name, "operator")


def _operator_x_axis(keys, operator_x_param):
    from loguru import logger
    if operator_x_param is None:
        return list(range(len(keys))), list(keys), "instance", "Instance"
    pairs = []
    for k in keys:
        v = parse_key_param(k, operator_x_param)
        if v is None:
            logger.warning(f"Key '{k}' has no numeric '{operator_x_param}' token; skipping.")
            continue
        pairs.append((v, k))
    if not pairs:
        raise ValueError(f"No keys contained a numeric '{operator_x_param}' token.")
    pairs.sort(key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs], operator_x_param, f"${operator_x_param}$"


def _run_analytic_operator(qubit_operator, *, extremum, operator_x_param,
                           log_dir, plot_dir, hide_plot):
    if extremum not in ("min", "max"):
        raise ValueError(f"extremum must be 'min' or 'max'; got {extremum!r}.")
    path, pattern = parse_operator_spec(qubit_operator)
    keys = list_hamlib_keys(path, pattern)
    if not keys:
        raise ValueError(f"No Hamiltonian datasets found in '{path}'.")
    x_vals, keys, x_param, x_label = _operator_x_axis(keys, operator_x_param)

    Z = np.full((len(keys),), np.nan)
    for ix, key in enumerate(keys):
        Z[ix] = analytic_operator(load_hamlib_operator(path, key), extremum)

    model_name = os.path.splitext(os.path.basename(path))[0]
    plot_format = "2d"
    tag = _file_tag("analytic", plot_format, x_param, None)

    log_path = None
    if log_dir is not None:
        subdir = _operator_subdir(log_dir, model_name)
        os.makedirs(subdir, exist_ok=True)
        log_path = os.path.join(subdir, f"{tag}.json")
        log_data = {
            "type": "analytic",
            "plot_format": plot_format,
            "band_structure": False,
            "observable": "E",
            "parameters": {
                "model": model_name,
                "lattice": None,
                "qubit_operator": path,
                "keys": keys,
                "extremum": extremum,
                "model_params": {},
            },
            "x_param": x_param,
            "y_param": None,
            "x_values": x_vals,
            "y_values": [],
            "result": {"analytic": {ix: float(Z[ix]) for ix in range(len(keys))}},
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        subdir = _operator_subdir(plot_dir, model_name)
        os.makedirs(subdir, exist_ok=True)
        plot_path = os.path.join(subdir, f"{tag}.pdf")

    if plot_path is not None or not hide_plot:
        plot_analytic(
            x_vals, [], x_label, "$E$", Z,
            plot_format=plot_format, output_path=plot_path, hide_plot=hide_plot,
            x_is_momentum=False, y_is_momentum=False, z_label="$E$",
        )

    return AnalyticResult(
        model_name=model_name, lattice=None, x_param=x_param, y_param=None,
        x_values=x_vals, y_values=[], energies=Z, plot_format=plot_format,
        band_structure=False, log_path=log_path, plot_path=plot_path, _model_params={},
    )


def run_analytic(
    model=None,
    *,
    lattice=None,
    x_param: str | None = None,
    x_range=None,
    y_param: str | None = None,
    y_range=None,
    n_occ: int | None = None,
    model_params: dict | None = None,
    observable: str = "E",
    qubit_operator: str | None = None,
    extremum: str = "min",
    operator_x_param: str | None = None,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    heatmap: bool = False,
) -> AnalyticResult:
    if qubit_operator is not None:
        return _run_analytic_operator(
            qubit_operator, extremum=extremum, operator_x_param=operator_x_param,
            log_dir=log_dir, plot_dir=plot_dir, hide_plot=hide_plot,
        )
    model = _resolve_model(model)
    _ = model._build_H_matrix
    _ = model.get_observable(observable)

    x_param, x_range, y_param, y_range, is_1d = _normalize_sweep_axes(x_param, x_range, y_param, y_range)
    if heatmap and is_1d:
        raise ValueError("heatmap=True requires both x and y sweep axes; provide y_param/y_range.")
    _gate_momentum(model, x_param, y_param)

    is_band_structure_run = _is_band_structure_axes(model, x_param, y_param)
    if is_band_structure_run:
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

    is_band_structure = (x_kind == "momentum") or (y_kind == "momentum")
    if is_band_structure:
        if x_kind == "n_occ" or y_kind == "n_occ" or n_occ is not None:
            raise ValueError("n_occ is not meaningful for band-structure (momentum-space) runs.")
        _ = model.bloch_hamiltonian
        for k_axis in momentum_axes:
            if k_axis not in (x_param, y_param) and k_axis not in params:
                raise ValueError(
                    f"Model '{model.name}' has momentum axis '{k_axis}' that is neither the active "
                    f"sweep axis nor fixed in model_params. Pin it explicitly to a value."
                )

    x_label = _label_for(model, x_param)
    obs_label = _observable_label(model, observable)
    y_label = _label_for(model, y_param) if not is_1d else obs_label

    plot_format = "2d" if is_1d else ("heatmap" if heatmap else "3d")

    if is_band_structure:
        probe_k = []
        for a in momentum_axes:
            if a == x_param:
                probe_k.append(x_vals[0])
            elif a == y_param:
                probe_k.append(y_vals[0])
            else:
                probe_k.append(params[a])
        probe_params = {k: v for k, v in params.items() if k not in momentum_axes}
        n_bands = model.bloch_hamiltonian(*probe_k, **probe_params).shape[0]
        Z_shape = (len(x_vals), n_bands) if is_1d else (len(x_vals), len(y_vals), n_bands)
    else:
        n_bands = 1
        Z_shape = (len(x_vals),) if is_1d else (len(x_vals), len(y_vals))
    Z = np.full(Z_shape, np.nan)

    def _eval_cell(xv, yv=None):
        cell_params = params.copy()
        n_occ_val = fixed_n_occ
        if x_kind == "n_occ":
            n_occ_val = int(xv)
        else:
            cell_params[x_param] = xv
        if yv is not None:
            if y_kind == "n_occ":
                n_occ_val = int(yv)
            else:
                cell_params[y_param] = yv
        if is_band_structure:
            k_tuple = tuple(cell_params.pop(a) for a in momentum_axes)
            return analytic_bands(model, k_tuple, cell_params, observable=observable)
        return analytic(model, lattice, n_occ_val, cell_params, observable=observable)

    if is_1d:
        for ix, xv in enumerate(x_vals):
            Z[ix] = _eval_cell(xv)
    else:
        for ix, xv in enumerate(x_vals):
            for iy, yv in enumerate(y_vals):
                Z[ix, iy] = _eval_cell(xv, yv)

    if is_1d:
        if is_band_structure:
            analytic_block = {ix: Z[ix, :].tolist() for ix in range(len(x_vals))}
        else:
            analytic_block = {ix: float(Z[ix]) for ix in range(len(x_vals))}
    else:
        if is_band_structure:
            analytic_block = {ix: {iy: Z[ix, iy, :].tolist() for iy in range(len(y_vals))} for ix in range(len(x_vals))}
        else:
            analytic_block = {ix: {iy: float(Z[ix, iy]) for iy in range(len(y_vals))} for ix in range(len(x_vals))}

    tag = _file_tag("analytic", plot_format, x_param, y_param, observable=observable)

    log_path = None
    if log_dir is not None:
        subdir = _log_subdir(log_dir, model.name, lattice)
        os.makedirs(subdir, exist_ok=True)
        log_path = os.path.join(subdir, f"{tag}.json")
        log_data = {
            "type": "analytic",
            "plot_format": plot_format,
            "band_structure": is_band_structure,
            "observable": observable,
            "parameters": {
                "model": model.name,
                "lattice": list(lattice) if lattice is not None else None,
                "model_params": {k: float(v) for k, v in params.items()},
            },
            "x_param": x_param,
            "y_param": y_param,
            "x_values": x_vals,
            "y_values": y_vals,
            "result": {"analytic": analytic_block},
        }
        if is_band_structure:
            log_data["n_bands"] = n_bands
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        subdir = _plot_subdir(plot_dir, model.name, lattice)
        os.makedirs(subdir, exist_ok=True)
        plot_path = os.path.join(subdir, f"{tag}.pdf")

    if plot_path is not None or not hide_plot:
        plot_analytic(
            x_vals, y_vals, x_label, y_label, Z,
            plot_format=plot_format,
            output_path=plot_path, hide_plot=hide_plot,
            x_is_momentum=(x_kind == "momentum"),
            y_is_momentum=(y_kind == "momentum"),
            z_label=obs_label,
        )

    return AnalyticResult(
        model_name=model.name,
        lattice=lattice,
        x_param=x_param,
        y_param=y_param,
        x_values=x_vals,
        y_values=y_vals,
        energies=Z,
        plot_format=plot_format,
        band_structure=is_band_structure,
        log_path=log_path,
        plot_path=plot_path,
        _model_params=params,
    )


def _run_simulated(
    model: Model,
    simulation_tag: str,
    backend,
    *,
    lattice: tuple[int, ...],
    x_param: str,
    x_range,
    y_param: str | None,
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
    is_1d: bool,
    observable: str = "E",
) -> SimulatedResult:
    do_vqe = vqe_reps > 0
    do_iqpe = iqpe_reps > 0

    obs = model.get_observable(observable)
    iqpe_supports_observable = (
        observable == "E"
        or (obs.quantum_composite is not None and observable == "charge_gap")
    )
    if do_iqpe and not iqpe_supports_observable:
        from loguru import logger as _logger
        _logger.warning(
            f"IQPE cannot measure observable '{observable}' directly; "
            f"only the analytic and VQE backends will be computed."
        )
        do_iqpe = False
        iqpe_reps = 0

    spin = model.spin
    if lattice is not None:
        n_sites = math.prod(lattice) * model.sites_per_cell
        n_orbitals = n_sites * spin
    else:
        n_sites = 0
        n_orbitals = 0
    fixed_n_occ = n_occ if n_occ is not None else (n_orbitals // 2 if n_orbitals else 0)
    mapper = model.get_mapper(n_sites, spin, fixed_n_occ)
    momentum_axes = model.momentum_axes

    x_vals, _x_label_default, x_kind = resolve_sweep(x_param, x_range, n_orbitals, momentum_axes)
    if is_1d:
        y_vals, y_kind = [], "none"
    else:
        y_vals, _y_label_default, y_kind = resolve_sweep(y_param, y_range, n_orbitals, momentum_axes)
    nx = len(x_vals)
    ny = 1 if is_1d else len(y_vals)
    plot_format = "2d" if is_1d else "3d"

    is_band_structure = (x_kind == "momentum") or (y_kind == "momentum")
    if is_band_structure:
        if x_kind == "n_occ" or y_kind == "n_occ" or n_occ is not None:
            raise ValueError("n_occ is not meaningful for band-structure (momentum-space) runs.")
        _ = model.bloch_hamiltonian
        for k_axis in momentum_axes:
            if k_axis not in (x_param, y_param) and k_axis not in model_params:
                raise ValueError(
                    f"Model '{model.name}' has momentum axis '{k_axis}' that is neither the active "
                    f"sweep axis nor fixed in model_params. Pin it explicitly to a value."
                )

    x_label = _label_for(model, x_param)
    y_label = _label_for(model, y_param) if not is_1d else "$E$"

    def cell_params_and_nocc(ix, iy):
        cp = model_params.copy()
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

    def split_k_and_params(cp):
        k_tuple = tuple(cp.pop(a) for a in momentum_axes)
        return k_tuple, cp

    def tagged_job(tag, func, *a, **kw):
        return tag, func(*a, **kw)

    jobs = []
    for ix in range(nx):
        for iy in range(ny):
            cp, n_occ_val = cell_params_and_nocc(ix, iy)
            if is_band_structure:
                k_tuple, cp_no_k = split_k_and_params(cp)
                jobs.append(delayed(tagged_job)(
                    ("analytic", ix, iy), analytic_bands, model, k_tuple, cp_no_k
                ))
                if do_iqpe:
                    for rep in range(1, iqpe_reps + 1):
                        jobs.append(delayed(tagged_job)(
                            ("iqpe", ix, iy, rep), iqpe_bloch,
                            k_tuple, cp_no_k, model.bloch_hamiltonian,
                            iqpe_time, iqpe_trot, iqpe_iters, rep,
                            backend=backend
                        ))
                    jobs.append(delayed(tagged_job)(
                        ("iqpe_bench", ix, iy), iqpe_bloch_other_benchmarks,
                        k_tuple, cp_no_k, model.bloch_hamiltonian,
                        iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps,
                        backend=backend
                    ))
                if do_vqe:
                    for rep in range(1, vqe_reps + 1):
                        jobs.append(delayed(tagged_job)(
                            ("vqe", ix, iy, rep), vqe_bloch,
                            k_tuple, cp_no_k, model.bloch_hamiltonian, model.get_optimizer,
                            vqe_iters, vqe_layers, rep,
                            backend=backend
                        ))
                    jobs.append(delayed(tagged_job)(
                        ("vqe_bench", ix, iy), vqe_bloch_other_benchmarks,
                        k_tuple, cp_no_k, model.bloch_hamiltonian,
                        vqe_iters, vqe_layers, vqe_reps,
                        backend=backend
                    ))
                continue

            jobs.append(delayed(tagged_job)(
                ("analytic", ix, iy), analytic, model, lattice, n_occ_val, cp, observable
            ))
            if do_iqpe:
                for rep in range(1, iqpe_reps + 1):
                    if observable == "E":
                        jobs.append(delayed(tagged_job)(
                            ("iqpe", ix, iy, rep), iqpe_fermionic,
                            lattice, n_sites, spin, n_occ_val, cp, model.fermionic_hamiltonian,
                            mapper, iqpe_time, iqpe_trot, iqpe_iters, rep,
                            backend=backend
                        ))
                    else:
                        jobs.append(delayed(tagged_job)(
                            ("iqpe", ix, iy, rep), iqpe_observable,
                            model, lattice, n_sites, spin, n_occ_val, cp,
                            mapper, iqpe_time, iqpe_trot, iqpe_iters, rep, observable,
                            backend=backend
                        ))
                jobs.append(delayed(tagged_job)(
                    ("iqpe_bench", ix, iy), iqpe_other_benchmarks,
                    lattice, n_sites, spin, n_occ_val, cp, model.fermionic_hamiltonian,
                    mapper, iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps,
                    backend=backend
                ))
            if do_vqe:
                for rep in range(1, vqe_reps + 1):
                    if observable == "E":
                        jobs.append(delayed(tagged_job)(
                            ("vqe", ix, iy, rep), vqe_fermionic,
                            lattice, n_sites, spin, n_occ_val, cp, model.fermionic_hamiltonian, model.get_optimizer,
                            model.get_vqe_ansatz,
                            mapper, vqe_iters, vqe_layers, rep,
                            backend=backend
                        ))
                    else:
                        jobs.append(delayed(tagged_job)(
                            ("vqe", ix, iy, rep), vqe_observable,
                            model, lattice, n_sites, spin, n_occ_val, cp,
                            mapper, vqe_iters, vqe_layers, rep, observable,
                            backend=backend
                        ))
                jobs.append(delayed(tagged_job)(
                    ("vqe_bench", ix, iy), vqe_other_benchmarks,
                    lattice, n_sites, spin, n_occ_val, cp, model.fermionic_hamiltonian,
                    model.get_vqe_ansatz,
                    mapper, vqe_iters, vqe_layers, vqe_reps,
                    backend=backend
                ))

    raw_tag = _file_tag(f"simulated-{simulation_tag}", plot_format, x_param, y_param, observable=observable)
    raw_data_path = None
    if log_dir is not None:
        log_subdir = _log_subdir(log_dir, model.name, lattice)
        os.makedirs(os.path.join(log_subdir, "raw-data"), exist_ok=True)
        raw_data_path = os.path.join(log_subdir, "raw-data", f"{raw_tag}.json")

    def empty_cell():
        cell = {"analytic": None}
        if do_vqe:
            cell["vqe"] = {"repetitions": [], "num_queries": None, "circuit_depth": None}
        if do_iqpe:
            cell["iqpe"] = {"repetitions": [], "iteration_energies": [], "num_queries": None, "circuit_depth": None}
        return cell

    parameters = {
        "model": model.name,
        "lattice": list(lattice),
        "simulation": simulation_tag,
        "model_params": {k: float(v) for k, v in model_params.items()},
    }
    if do_vqe:
        parameters["vqe"] = {"iters": vqe_iters, "layers": vqe_layers, "reps": vqe_reps}
    if do_iqpe:
        parameters["iqpe"] = {"time": iqpe_time, "trot": iqpe_trot, "iters": iqpe_iters, "reps": iqpe_reps}

    raw_data = {
        "parameters": parameters,
        "plot_format": plot_format,
        "band_structure": is_band_structure,
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "grid": {
            str(ix): {str(iy): empty_cell() for iy in range(ny)}
            for ix in range(nx)
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

    if is_band_structure:
        probe_k = []
        for a in momentum_axes:
            if a == x_param:
                probe_k.append(x_vals[0])
            elif a == y_param:
                probe_k.append(y_vals[0])
            else:
                probe_k.append(model_params[a])
        probe_params = {k: v for k, v in model_params.items() if k not in momentum_axes}
        n_bands = model.bloch_hamiltonian(*probe_k, **probe_params).shape[0]
        Z_exact_bands = np.full((nx, ny, n_bands), np.nan)
        Z_exact = np.full((nx, ny), np.nan)
    else:
        n_bands = 1
        Z_exact_bands = None
        Z_exact = np.full((nx, ny), np.nan)
    Z_vqe = np.full((nx, ny), np.nan) if do_vqe else None
    Z_iqpe = np.full((nx, ny), np.nan) if do_iqpe else None

    for ix in range(nx):
        for iy in range(ny):
            cell = raw_data["grid"][str(ix)][str(iy)]
            if is_band_structure:
                bands = np.array(cell["analytic"], dtype=float)
                Z_exact_bands[ix, iy, :] = bands
                Z_exact[ix, iy] = bands[0]
            else:
                Z_exact[ix, iy] = cell["analytic"]
            loc = f"{x_param}={x_vals[ix]}" + ("" if is_1d else f", {y_param}={y_vals[iy]}")
            if do_iqpe:
                Z_iqpe[ix, iy] = min(cell["iqpe"]["repetitions"], key=lambda e: abs(e - Z_exact[ix, iy]))
                logger.info(f"IQPE ({loc}) = {Z_iqpe[ix, iy]}")
            if do_vqe:
                Z_vqe[ix, iy] = min(cell["vqe"]["repetitions"], key=lambda e: abs(e - Z_exact[ix, iy]))
                logger.info(f"VQE  ({loc}) = {Z_vqe[ix, iy]}")

    def _cell_scalar(arr, ix, iy):
        return float(arr[ix, iy])

    def _cell_bands(arr, ix, iy):
        return arr[ix, iy, :].tolist()

    def _build_block(arr, *, bands: bool):
        getter = _cell_bands if bands else _cell_scalar
        if is_1d:
            return {ix: getter(arr, ix, 0) for ix in range(nx)}
        return {ix: {iy: getter(arr, ix, iy) for iy in range(ny)} for ix in range(nx)}

    if is_band_structure:
        analytic_block = _build_block(Z_exact_bands, bands=True)
    else:
        analytic_block = _build_block(Z_exact, bands=False)
    result_block = {"analytic": analytic_block}

    num_queries_block = {}
    depth_total_block = {}
    depth_two_q_block = {}

    def _bench_block(method: str, key: str):
        if is_1d:
            return {ix: raw_data["grid"][str(ix)]["0"][method][key] for ix in range(nx)}
        return {ix: {iy: raw_data["grid"][str(ix)][str(iy)][method][key] for iy in range(ny)} for ix in range(nx)}

    def _depth_block(method: str, sub: str):
        def get(ix, iy):
            return raw_data["grid"][str(ix)][str(iy)][method]["circuit_depth"][sub]
        if is_1d:
            return {ix: get(ix, 0) for ix in range(nx)}
        return {ix: {iy: get(ix, iy) for iy in range(ny)} for ix in range(nx)}

    if do_iqpe:
        result_block["iqpe"] = _build_block(Z_iqpe, bands=False)
        num_queries_block["iqpe"] = _bench_block("iqpe", "num_queries")
        depth_total_block["iqpe"] = _depth_block("iqpe", "total")
        depth_two_q_block["iqpe"] = _depth_block("iqpe", "two_qubit")
    if do_vqe:
        result_block["vqe"] = _build_block(Z_vqe, bands=False)
        num_queries_block["vqe"] = _bench_block("vqe", "num_queries")
        depth_total_block["vqe"] = _depth_block("vqe", "total")
        depth_two_q_block["vqe"] = _depth_block("vqe", "two_qubit")

    summary = {
        "type": f"simulated-{simulation_tag}",
        "plot_format": plot_format,
        "band_structure": is_band_structure,
        "parameters": raw_data["parameters"],
        "x_param": x_param, "y_param": y_param,
        "x_values": x_vals, "y_values": y_vals,
        "result": result_block,
    }
    if is_band_structure:
        summary["n_bands"] = n_bands
    if do_vqe or do_iqpe:
        summary["num_queries"] = num_queries_block
        summary["circuit_depth"] = {"total": depth_total_block, "two_qubit": depth_two_q_block}

    summary_path = None
    if log_dir is not None:
        summary_path = os.path.join(log_subdir, f"{raw_tag}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        plot_subdir = _plot_subdir(plot_dir, model.name, lattice)
        os.makedirs(plot_subdir, exist_ok=True)
        plot_path = os.path.join(plot_subdir, f"{raw_tag}.pdf")

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


def _run_simulated_operator(qubit_operator, simulation_tag, backend, *, extremum, operator_x_param,
                            ansatz, optimizer,
                            vqe_iters, vqe_layers, vqe_reps,
                            iqpe_time, iqpe_trot, iqpe_iters, iqpe_reps,
                            log_dir, plot_dir, hide_plot, hide_legend):
    from loguru import logger
    from quaph._yaml_model import (
        AnsatzSpec, OptimizerSpec, build_ansatz_factory, build_optimizer_factory,
    )

    if extremum not in ("min", "max"):
        raise ValueError(f"extremum must be 'min' or 'max'; got {extremum!r}.")

    do_vqe = vqe_reps > 0
    do_iqpe = iqpe_reps > 0

    path, pattern = parse_operator_spec(qubit_operator)
    keys = list_hamlib_keys(path, pattern)
    if not keys:
        raise ValueError(f"No Hamiltonian datasets found in '{path}'.")
    x_vals, keys, x_param, x_label = _operator_x_axis(keys, operator_x_param)
    nx = len(keys)
    model_name = os.path.splitext(os.path.basename(path))[0]

    if do_vqe:
        ansatz_spec = AnsatzSpec.model_validate(ansatz) if ansatz else AnsatzSpec(
            type="efficient_su2", kwargs={"reps": "@n_layers"}, initial_state_prefix="none",
        )
        get_vqe_ansatz = build_ansatz_factory(ansatz_spec, name="operator")
        optimizer_spec = OptimizerSpec.model_validate(optimizer) if optimizer else OptimizerSpec(
            type="SPSA", kwargs={"maxiter": "@max_iters"},
        )
        get_optimizer = build_optimizer_factory(optimizer_spec, name="operator")

    ops = [load_hamlib_operator(path, k) for k in keys]

    def tagged_job(tag, func, *a, **kw):
        return tag, func(*a, **kw)

    jobs = []
    for ix in range(nx):
        op = ops[ix]
        jobs.append(delayed(tagged_job)(("analytic", ix), analytic_operator, op, extremum))
        if do_iqpe:
            for rep in range(1, iqpe_reps + 1):
                jobs.append(delayed(tagged_job)(
                    ("iqpe", ix, rep), iqpe_operator,
                    op, iqpe_time, iqpe_trot, iqpe_iters, rep, extremum, backend,
                ))
        if do_vqe:
            for rep in range(1, vqe_reps + 1):
                jobs.append(delayed(tagged_job)(
                    ("vqe", ix, rep), vqe_operator,
                    op, get_vqe_ansatz, get_optimizer, vqe_iters, vqe_layers, rep, extremum, backend,
                ))

    grid = {ix: {"analytic": None, "vqe": [], "iqpe": [], "iqpe_iters": []} for ix in range(nx)}

    def init_worker_logging():
        from quaph._core import setup_logging as _sl
        _sl()

    for tag, result in Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(jobs):
        ix = tag[1]
        if tag[0] == "analytic":
            grid[ix]["analytic"] = result
        elif tag[0] == "vqe":
            grid[ix]["vqe"].append(result)
        elif tag[0] == "iqpe":
            energy, iter_energies = result
            grid[ix]["iqpe"].append(energy)
            grid[ix]["iqpe_iters"].append(iter_energies)

    Z_exact = np.array([grid[ix]["analytic"] for ix in range(nx)], dtype=float)
    Z_vqe = np.full(nx, np.nan) if do_vqe else None
    Z_iqpe = np.full(nx, np.nan) if do_iqpe else None
    for ix in range(nx):
        loc = f"{x_param}={x_vals[ix]}"
        if do_iqpe:
            Z_iqpe[ix] = min(grid[ix]["iqpe"], key=lambda e: abs(e - Z_exact[ix]))
            logger.info(f"IQPE ({loc}) = {Z_iqpe[ix]}")
        if do_vqe:
            Z_vqe[ix] = min(grid[ix]["vqe"], key=lambda e: abs(e - Z_exact[ix]))
            logger.info(f"VQE  ({loc}) = {Z_vqe[ix]}")

    plot_format = "2d"
    parameters = {
        "model": model_name,
        "lattice": None,
        "qubit_operator": path,
        "keys": keys,
        "extremum": extremum,
        "simulation": simulation_tag,
        "model_params": {},
    }
    if do_vqe:
        parameters["vqe"] = {"iters": vqe_iters, "layers": vqe_layers, "reps": vqe_reps}
    if do_iqpe:
        parameters["iqpe"] = {"time": iqpe_time, "trot": iqpe_trot, "iters": iqpe_iters, "reps": iqpe_reps}

    result_block = {"analytic": {ix: float(Z_exact[ix]) for ix in range(nx)}}
    if do_iqpe:
        result_block["iqpe"] = {ix: float(Z_iqpe[ix]) for ix in range(nx)}
    if do_vqe:
        result_block["vqe"] = {ix: float(Z_vqe[ix]) for ix in range(nx)}

    summary = {
        "type": f"simulated-{simulation_tag}",
        "plot_format": plot_format,
        "band_structure": False,
        "parameters": parameters,
        "x_param": x_param, "y_param": None,
        "x_values": x_vals, "y_values": [],
        "result": result_block,
    }

    tag = _file_tag(f"simulated-{simulation_tag}", plot_format, x_param, None)
    summary_path = None
    if log_dir is not None:
        subdir = _operator_subdir(log_dir, model_name)
        os.makedirs(subdir, exist_ok=True)
        summary_path = os.path.join(subdir, f"{tag}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    plot_path = None
    if plot_dir is not None:
        subdir = _operator_subdir(plot_dir, model_name)
        os.makedirs(subdir, exist_ok=True)
        plot_path = os.path.join(subdir, f"{tag}.pdf")

    if plot_path is not None or not hide_plot:
        plot_simulated(
            x_vals, [], x_label, "$E$", Z_exact, Z_vqe, Z_iqpe,
            plot_format=plot_format, hide_legend=hide_legend,
            output_path=plot_path, hide_plot=hide_plot,
            x_is_momentum=False, y_is_momentum=False,
        )

    return SimulatedResult(
        model_name=model_name, lattice=None, x_param=x_param, y_param=None,
        x_values=x_vals, y_values=[], analytic_energies=Z_exact,
        vqe_best_energies=Z_vqe, iqpe_best_energies=Z_iqpe,
        plot_format=plot_format, band_structure=False, analytic_bands=None,
        raw=summary, raw_log_path=None, summary_log_path=summary_path,
        plot_path=plot_path, _model_params={},
    )


def run_simulated_ideal(
    model=None,
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
    qubit_operator: str | None = None,
    extremum: str = "min",
    operator_x_param: str | None = None,
    ansatz: dict | None = None,
    optimizer: dict | None = None,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
    observable: str = "E",
) -> SimulatedResult:
    if qubit_operator is not None:
        vqe_reps = _resolve_method_reps("vqe", vqe_reps, vqe_iters, vqe_layers)
        iqpe_reps = _resolve_method_reps("iqpe", iqpe_reps, iqpe_time, iqpe_trot, iqpe_iters)
        return _run_simulated_operator(
            qubit_operator, "ideal", None,
            extremum=extremum, operator_x_param=operator_x_param,
            ansatz=ansatz, optimizer=optimizer,
            vqe_iters=vqe_iters, vqe_layers=vqe_layers, vqe_reps=vqe_reps,
            iqpe_time=iqpe_time, iqpe_trot=iqpe_trot, iqpe_iters=iqpe_iters, iqpe_reps=iqpe_reps,
            log_dir=log_dir, plot_dir=plot_dir, hide_plot=hide_plot, hide_legend=hide_legend,
        )

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
        is_1d=is_1d, observable=observable,
    )


def run_simulated_noisy(
    model=None,
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
    qubit_operator: str | None = None,
    extremum: str = "min",
    operator_x_param: str | None = None,
    ansatz: dict | None = None,
    optimizer: dict | None = None,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    hide_legend: bool = False,
    observable: str = "E",
) -> SimulatedResult:
    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()

    if qubit_operator is not None:
        vqe_reps = _resolve_method_reps("vqe", vqe_reps, vqe_iters, vqe_layers)
        iqpe_reps = _resolve_method_reps("iqpe", iqpe_reps, iqpe_time, iqpe_trot, iqpe_iters)
        return _run_simulated_operator(
            qubit_operator, "noisy", backend,
            extremum=extremum, operator_x_param=operator_x_param,
            ansatz=ansatz, optimizer=optimizer,
            vqe_iters=vqe_iters, vqe_layers=vqe_layers, vqe_reps=vqe_reps,
            iqpe_time=iqpe_time, iqpe_trot=iqpe_trot, iqpe_iters=iqpe_iters, iqpe_reps=iqpe_reps,
            log_dir=log_dir, plot_dir=plot_dir, hide_plot=hide_plot, hide_legend=hide_legend,
        )

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
        is_1d=is_1d, observable=observable,
    )
