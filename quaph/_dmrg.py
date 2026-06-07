from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from quaph._core import analytic, resolve_sweep
from quaph._model import Model


@dataclass
class DMRGRun:
    n_occ: int
    model_params: dict
    energy: float
    output_path: str
    raw: dict


def _lattice_tag(lattice) -> str:
    return "x".join(str(x) for x in lattice)


def _file_tag(x_param, y_param):
    if y_param is None:
        return f"dmrg-{x_param}"
    return f"dmrg-{x_param}-vs-{y_param}"


def _complex_payload(value):
    c = complex(value)
    return {"re": c.real, "im": c.imag}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _export_fermionic_op(model: Model, lattice, n_occ: int, model_params: dict, path: str):
    op = model.fermionic_hamiltonian(lattice, **model_params)
    terms = []
    for label, coeff in op.items():
        terms.append({"label": label, "coefficient": _complex_payload(coeff)})

    payload = {
        "format": "quaph_fermionic_op_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model.name,
        "display_name": model.display_name,
        "lattice": list(lattice),
        "spin": model.spin,
        "n_sites": int(op.num_spin_orbitals // model.spin),
        "num_spin_orbitals": int(op.num_spin_orbitals),
        "n_occ": int(n_occ),
        "model_params": model_params,
        "terms": terms,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _default_julia_script():
    return Path(__file__).resolve().parent.parent / "scripts" / "julia-dmrg" / "dmrg_itensor_cli.jl"


def _default_julia_project():
    return Path(__file__).resolve().parent.parent / "scripts" / "julia-dmrg"


def _default_julia_depot():
    return Path(os.environ.get("QUAPH_JULIA_DEPOT", "/pscratch/sd/m/mbao202/julia_depot"))


def _nersc_julia_path():
    path = Path("/global/common/software/nersc9/julia/1.11.7/bin/julia")
    return str(path) if path.exists() else None


def _run_julia_dmrg(
    hamiltonian_path: str,
    output_path: str,
    *,
    julia: str,
    julia_module: str | None,
    julia_project: str | None,
    nsweeps: int,
    maxdims: str,
    cutoff: float,
    seed: int,
    conserve_qns: bool,
    script_path: str | None,
):
    script = str(Path(script_path) if script_path else _default_julia_script())
    project = julia_project or str(_default_julia_project())
    julia_args = [
        f"--project={project}",
        script,
        "--hamiltonian",
        hamiltonian_path,
        "--output",
        output_path,
        "--nsweeps",
        str(nsweeps),
        "--maxdims",
        maxdims,
        "--cutoff",
        str(cutoff),
        "--seed",
        str(seed),
        "--conserve-qns",
        "true" if conserve_qns else "false",
    ]
    resolved_julia = shutil.which(julia)
    if resolved_julia is None and julia == "julia":
        resolved_julia = _nersc_julia_path()
    if resolved_julia is not None:
        cmd = [resolved_julia, *julia_args]
    elif os.path.isabs(julia):
        cmd = [julia, *julia_args]
    elif julia_module:
        module_cmd = f"module load {shlex.quote(julia_module)} && exec {shlex.quote(julia)} \"$@\""
        cmd = ["bash", "-lc", module_cmd, "quaph-julia", *julia_args]
    else:
        cmd = [julia, *julia_args]
    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", str(_default_julia_depot()))
    subprocess.run(cmd, check=True, env=env)


def _cell_params_and_nocc(
    x_param,
    y_param,
    x_values,
    y_values,
    ix,
    iy,
    fixed_n_occ,
    base_params,
):
    params = dict(base_params)
    n_occ = fixed_n_occ
    xv = x_values[ix]
    if x_param == "n_occ":
        n_occ = int(xv)
    else:
        params[x_param] = float(xv)

    if y_param is not None:
        yv = y_values[iy]
        if y_param == "n_occ":
            n_occ = int(yv)
        else:
            params[y_param] = float(yv)

    return params, int(n_occ)


def run_dmrg_itensor(
    model: Model,
    *,
    lattice,
    x_param: str | None,
    x_range,
    y_param: str | None,
    y_range,
    n_occ: int | None,
    model_params: dict,
    log_dir=None,
    plot_dir=None,
    hide_plot: bool = False,
    julia: str = "julia",
    julia_module: str | None = "julia/1.11.7",
    julia_project: str | None = None,
    nsweeps: int = 4,
    maxdims: str = "20,50,100,200",
    cutoff: float = 1e-9,
    seed: int = 1234,
    conserve_qns: bool = True,
    script_path: str | None = None,
    task_index: int | None = None,
    task_count: int = 1,
    prepare_only: bool = False,
    aggregate_only: bool = False,
    no_progress_log: bool = False,
) -> dict:
    if lattice is None:
        raise ValueError("DMRG requires a real-space --lattice.")
    lattice = tuple(int(x) for x in lattice)
    n_sites = int(np.prod(lattice) * model.sites_per_cell)
    n_orbitals = n_sites * model.spin
    fixed_n_occ = n_occ if n_occ is not None else n_orbitals // 2

    if x_param is None:
        x_param = "n_occ"
    x_values, _, _ = resolve_sweep(x_param, x_range, n_orbitals, model.momentum_axes)
    if y_param is None:
        y_values = [None]
    else:
        y_values, _, _ = resolve_sweep(y_param, y_range, n_orbitals, model.momentum_axes)

    if x_param in model.momentum_axes or y_param in model.momentum_axes:
        raise ValueError("DMRG only supports real-space parameter and n_occ sweeps.")
    if task_count < 1:
        raise ValueError("task_count must be at least 1")
    if task_index is not None and not 0 <= task_index < task_count:
        raise ValueError("task_index must satisfy 0 <= task_index < task_count")

    root = Path(log_dir or "logs")
    outdir = root / model.name / _lattice_tag(lattice) / "dmrg"
    raw_dir = outdir / "raw-data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / f"{_file_tag(x_param, y_param)}.json"
    progress_path = raw_dir / f"{_file_tag(x_param, y_param)}.progress.jsonl"
    plot_path = None
    if plot_dir is not None:
        plot_path = (
            Path(plot_dir)
            / model.name
            / _lattice_tag(lattice)
            / "dmrg"
            / f"{_file_tag(x_param, y_param)}.pdf"
        )

    cells = []
    for ix in range(len(x_values)):
        for iy in range(len(y_values)):
            params, cell_n_occ = _cell_params_and_nocc(
                x_param,
                y_param,
                x_values,
                y_values,
                ix,
                iy,
                fixed_n_occ,
                model_params,
            )
            cells.append((ix, iy, params, cell_n_occ))

    def empty_summary():
        return {
            "type": "dmrg",
            "algorithm": "itensor-mps-dmrg",
            "model": model.name,
            "lattice": list(lattice),
            "x_param": x_param,
            "x_values": _jsonable(x_values),
            "y_param": y_param,
            "y_values": [] if y_param is None else _jsonable(y_values),
            "parameters": {
                "model_params": _jsonable(model_params),
                "n_occ": n_occ,
                "nsweeps": nsweeps,
                "maxdims": maxdims,
                "cutoff": cutoff,
                "seed": seed,
                "conserve_qns": conserve_qns,
                "task_count": task_count,
            },
            "result": {str(ix): {} for ix in range(len(x_values))},
            "runs": [],
        }

    summary = empty_summary()
    exact_values = np.full((len(x_values), len(y_values)), np.nan)
    dmrg_values = np.full((len(x_values), len(y_values)), np.nan)

    def apply_cell(ix, iy, cell):
        summary["result"].setdefault(str(ix), {})[str(iy)] = cell
        summary["runs"].append(cell)
        dmrg_values[ix, iy] = float(cell["energy"])
        if cell["exact_energy"] is not None:
            exact_values[ix, iy] = float(cell["exact_energy"])

    def append_progress(ix, iy, cell):
        if no_progress_log:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ix": ix,
            "iy": iy,
            "cell": _jsonable(cell),
        }
        payload = (json.dumps(record) + "\n").encode()
        fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    def load_progress():
        if not progress_path.exists():
            raise FileNotFoundError(f"Progress file does not exist: {progress_path}")
        with open(progress_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                apply_cell(int(record["ix"]), int(record["iy"]), record["cell"])

    def compute_cell(ix, iy, params, cell_n_occ, tmp):
        cell_tag = f"x{ix}" if y_param is None else f"x{ix}-y{iy}"
        hamiltonian_path = os.path.join(tmp, f"{cell_tag}-hamiltonian.json")
        output_path = raw_dir / f"{cell_tag}.json"
        spec = _export_fermionic_op(model, lattice, cell_n_occ, params, hamiltonian_path)
        _run_julia_dmrg(
            hamiltonian_path,
            str(output_path),
            julia=julia,
            julia_module=julia_module,
            julia_project=julia_project,
            nsweeps=nsweeps,
            maxdims=maxdims,
            cutoff=cutoff,
            seed=seed + ix * len(y_values) + iy,
            conserve_qns=conserve_qns,
            script_path=script_path,
        )
        with open(output_path) as f:
            raw = json.load(f)
        energy = float(raw["energy"])
        try:
            exact_energy = float(analytic(model, lattice, cell_n_occ, params))
        except Exception:
            exact_energy = None
        return {
            "n_occ": int(cell_n_occ),
            "model_params": _jsonable(params),
            "energy": energy,
            "exact_energy": exact_energy,
            "error": None if exact_energy is None else energy - exact_energy,
            "output": str(output_path),
            "hamiltonian_terms": len(spec["terms"]),
            "max_link_dim": raw.get("max_link_dim"),
            "avg_link_dim": raw.get("avg_link_dim"),
            "profile": raw.get("profile"),
        }

    if prepare_only:
        if not no_progress_log:
            with open(progress_path, "w") as f:
                f.write("")
        with open(summary_path, "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        summary["summary_path"] = str(summary_path)
        return summary

    if aggregate_only:
        load_progress()
    else:
        selected = cells
        if task_index is not None:
            selected = [cell for idx, cell in enumerate(cells) if idx % task_count == task_index]
        with tempfile.TemporaryDirectory(prefix="quaph-dmrg-") as tmp:
            for ix, iy, params, cell_n_occ in selected:
                cell = compute_cell(ix, iy, params, cell_n_occ, tmp)
                apply_cell(ix, iy, cell)
                append_progress(ix, iy, cell)
        if task_index is not None:
            return {
                "type": "dmrg-shard",
                "task_index": task_index,
                "task_count": task_count,
                "num_cells": len(selected),
            }

    summary["runs"] = [
        summary["result"][str(ix)][str(iy)]
        for ix in range(len(x_values))
        for iy in range(len(y_values))
        if str(iy) in summary["result"].get(str(ix), {})
    ]
    expected = len(x_values) * len(y_values)
    if len(summary["runs"]) != expected:
        raise RuntimeError(f"Missing DMRG cells before aggregation: {len(summary['runs'])}/{expected}")

    if plot_path is not None and np.isfinite(exact_values).all():
        from quaph._plotting import plot_simulated

        is_1d = y_param is None
        plot_simulated(
            np.asarray(x_values, dtype=float),
            [] if is_1d else np.asarray(y_values, dtype=float),
            model.param_labels.get(x_param, x_param),
            "$E$" if is_1d else model.param_labels.get(y_param, y_param),
            exact_values[:, 0] if is_1d else exact_values,
            dmrg_values[:, 0] if is_1d else dmrg_values,
            None,
            plot_format="2d" if is_1d else "3d",
            output_path=str(plot_path),
            hide_plot=hide_plot,
            vqe_label="DMRG",
        )
        summary["plot_path"] = str(plot_path)
    with open(summary_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary
