"""Offline QPU-cost estimation for real IBM/IQM backends.

``qbp.estimate(...)`` mirrors :func:`qbp.run` but, instead of executing the
quantum methods, it compiles every circuit the run would submit and reports the
total estimated QPU execution time in seconds -- the unit both IBM (quantum
seconds) and IQM (time-based credits) meter. Nothing is submitted and no credits
are spent. It requires a real IBM or IQM backend; a local simulator (ideal Aer
or a fake backend) has no credit cost and raises instead.

The per-circuit cost is computed by :func:`qbp._backend.circuit_qpu_seconds`, and
the number of circuits per cell comes from each method's ``estimate_*_cell``
hook. The sweep grid, geometry and cell parameters are resolved with exactly the
same planning code the run path uses (:func:`qbp._run._plan_model_cells` and the
operator-axis helpers), so the estimate covers precisely the cells a run would.

Assumptions (the returned figure is an estimate, not a quote): VQE is assumed to
run its full ``iters x reps`` evaluations, each costing one circuit per commuting
Pauli group of the Hamiltonian. When ``shots`` is left unset each method uses the
same per-circuit shot count the real run would (VQE's estimator precision and
IQPE's sampler default: 1024 on IQM, 4096 on IBM); passing ``shots`` overrides
all paths uniformly. Classical methods (analytic/DMRG) never touch the device and
contribute nothing.
"""

from __future__ import annotations

from qbp._backend import (
    resolve_backend, backend_label as _backend_label, is_real_backend,
)
from qbp._boundary import _resolve_boundary
from qbp._hamlib import collect_keys_multi, load_hamlib_operator
from qbp._investigation import build_investigation
from qbp._method import CellContext
from qbp._run import (
    _normalize_methods, _build_method_objects, _resolve_model, _plan_model_cells,
    _flatten_op_dict, _resolve_operator_axes, _filter_keys,
    _diagnostic_kind_from_axes, _modified_fermionic_fn,
)


def estimate(
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
    shots: int | None = None,
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
) -> float:
    """Estimate the QPU cost of the equivalent :func:`qbp.run` call.

    Accepts the same arguments as :func:`qbp.run` (so any ``run`` invocation can
    be turned into an estimate by changing the verb), plus an optional ``shots``
    override. Left unset, each method uses the same per-circuit shot count the real
    run would. Output/plot/distribution arguments are accepted for signature parity
    but ignored. Prints and returns the total estimated QPU-seconds. Raises
    :class:`ValueError` unless ``backend`` resolves to a real IBM or IQM device.
    """
    shots = None if shots is None else int(shots)
    methods = _normalize_methods(method)
    boundary, bparams = _resolve_boundary(boundary, boundary_params)
    investigation = build_investigation(investigation, investigation_params)

    backend = resolve_backend(backend)
    if not is_real_backend(backend):
        raise ValueError(
            f"estimate() requires a real IBM or IQM backend; got "
            f"'{_backend_label(backend)}'. It compiles the run's circuits and reports "
            f"estimated QPU-seconds without executing, so a local simulator (ideal Aer "
            f"or a fake backend) has no credit cost to estimate -- use run() to simulate."
        )

    method_objs = _build_method_objects(methods, method_params)

    if qubit_operator is not None:
        if isinstance(qubit_operator, str):
            qubit_operator = [qubit_operator]
        elif isinstance(qubit_operator, dict):
            qubit_operator = _flatten_op_dict(qubit_operator)
        else:
            qubit_operator = list(qubit_operator)
        total = _estimate_operator(
            qubit_operator, methods, method_objs, backend,
            extremum=extremum, select=select, observable=observable,
            x_param=x_param, x_range=x_range, y_param=y_param, y_range=y_range,
            shots=shots,
        )
    else:
        total = _estimate_model(
            model, methods, method_objs, backend,
            lattice=lattice, x_param=x_param, x_range=x_range,
            y_param=y_param, y_range=y_range, n_occ=n_occ, model_params=model_params,
            boundary=boundary, geometry=bparams["geometry"], radius=bparams["radius"],
            center=bparams["center"], potential_profile=bparams["potential_profile"],
            potential_radius=bparams["potential_radius"], potential_v0=bparams["potential_v0"],
            potential_xi=bparams["potential_xi"], investigation=investigation,
            observable=observable, shots=shots,
        )

    print(total)
    return total


def _estimate_model(
    model, methods, method_objs, backend, *,
    lattice, x_param, x_range, y_param, y_range, n_occ, model_params,
    boundary, geometry, radius, center,
    potential_profile, potential_radius, potential_v0, potential_xi,
    investigation, observable, shots,
) -> float:
    if model is not None:
        diag_model = _resolve_model(model)
        if _diagnostic_kind_from_axes(diag_model, x_param, y_param) != "energy":
            return 0.0

    plan = _plan_model_cells(
        model, methods, method_objs,
        lattice=lattice, x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range, n_occ=n_occ, model_params=model_params,
        boundary=boundary, geometry=geometry, radius=radius, center=center,
        potential_profile=potential_profile, potential_radius=potential_radius,
        potential_v0=potential_v0, potential_xi=potential_xi,
        investigation=investigation, observable=observable,
    )

    total = 0.0
    for m in methods:
        m_obj = method_objs[m]
        for ix in range(plan.nx):
            for iy in range(plan.ny):
                cp, n_occ_val = plan.cell_params_and_nocc(ix, iy)
                ctx = CellContext(
                    ix=ix, iy=iy, cell_index=ix * plan.ny + iy,
                    n_sites=plan.n_sites, spin=plan.spin, n_orbitals=plan.n_orbitals,
                )
                if plan.is_band:
                    k_tuple = tuple(cp.pop(a) for a in plan.momentum_axes)
                    total += m_obj.estimate_bloch_cell(
                        plan.model, k_tuple, cp, observable,
                        backend=backend, ctx=ctx, shots=shots,
                    )
                    continue
                projected = plan.projection is not None and (
                    plan.projection.geometry != "rectangle" or plan.has_profiles
                )
                if projected:
                    ctx.fermionic_hamiltonian_fn = _modified_fermionic_fn(
                        plan.model, plan.projection, plan.potential_kwargs, investigation
                    )
                else:
                    ctx.fermionic_hamiltonian_fn = plan.model.fermionic_hamiltonian
                ctx.mapper = plan.model.get_mapper(plan.n_sites, plan.spin, n_occ_val)
                total += m_obj.estimate_cell(
                    plan.model, plan.lattice, n_occ_val, cp, observable,
                    backend=backend, ctx=ctx, shots=shots,
                )
    return total


def _estimate_operator(
    qubit_operator, methods, method_objs, backend, *,
    extremum, select, observable, x_param, x_range, y_param, y_range, shots,
) -> float:
    for m in methods:
        if not method_objs[m].SUPPORTS_OPERATOR:
            raise ValueError(
                f"Method '{m.value}' does not support the --qubit-operator path."
            )

    if isinstance(qubit_operator, dict):
        all_display_keys = list(qubit_operator.keys())
        if not all_display_keys:
            raise ValueError("qubit_operator dict is empty.")
        display_keys = _filter_keys(all_display_keys, select)

        def _load_op(display_key):
            return qubit_operator[display_key]
    else:
        paths = qubit_operator
        all_display_keys, key_to_source = collect_keys_multi(paths)
        if not all_display_keys:
            raise ValueError("No Hamiltonian datasets found in the provided source(s).")
        display_keys = _filter_keys(all_display_keys, select)

        def _load_op(display_key):
            file_path, actual_key = key_to_source[display_key]
            return load_hamlib_operator(file_path, actual_key)

    (x_param, x_vals, _x_label, y_param, y_vals, _y_label, key_grid,
     is_1d) = _resolve_operator_axes(display_keys, x_param, x_range, y_param, y_range)
    nx = len(x_vals)
    ny = 1 if is_1d else len(y_vals)

    total = 0.0
    for m in methods:
        m_obj = method_objs[m]
        for ix in range(nx):
            for iy in range(ny):
                display_key = key_grid[ix] if is_1d else key_grid[ix][iy]
                if display_key is None:
                    continue
                op = _load_op(display_key)
                total += m_obj.estimate_operator_cell(
                    op, extremum=extremum, backend=backend,
                    observable=observable, shots=shots,
                )
    return total
