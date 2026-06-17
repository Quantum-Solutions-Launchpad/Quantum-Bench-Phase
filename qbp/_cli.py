from __future__ import annotations

import argparse
import sys

from qbp._registry import get_model, register_model_from_file, remove_model
from qbp._run import run, load_result
from qbp._method import Method, METHOD_ORDER, get_method_class
from qbp._yaml_model import _QISKIT_ANSATZES, _QISKIT_OPTIMIZERS, _INITIAL_STATES


def _model_param_names(model) -> list[str]:
    momentum = set(model.momentum_axes)
    return [p for p in model.param_labels if p != "n_occ" and p not in momentum]


def _add_model_params(parser, model):
    for name in _model_param_names(model):
        parser.add_argument(
            f"--{name}", type=float, default=None, metavar=name.upper(),
            help=f"Value for model parameter {name} (required unless it is the active sweep axis)",
        )


def _collect_model_params(args, model, x_param, y_param) -> dict:
    params: dict = {}
    missing: list[str] = []
    for name in _model_param_names(model):
        val = getattr(args, name, None)
        if name == x_param or name == y_param:
            if val is not None:
                raise ValueError(
                    f"--{name} cannot be used as a fixed value while '{name}' is the active "
                    f"sweep axis. Override the sweep with --x-param/--y-param first."
                )
            continue
        if val is None:
            missing.append(name)
        else:
            params[name] = val
    if missing:
        raise ValueError(
            f"Missing required model parameters for '{model.name}': "
            f"{', '.join('--' + m for m in missing)}."
        )
    return params


def _add_sweep_args(parser):
    parser.add_argument("--x-param", default=None, metavar="PARAM",
                        help="Sweep axis. A model/operator parameter or n_occ runs an energy "
                             "sweep; a momentum axis (kx/ky) runs a band structure; the real-space "
                             "lattice axes (Lx/Ly) render a real-space eigenstate-density map, and "
                             "'eigenstate' renders an edge-participation spectrum.")
    parser.add_argument("--x-range", type=float, nargs="+", metavar="N", default=None,
                        help="MIN MAX [STEP]. Model sweeps require STEP; --qubit-operator "
                             "sweeps may omit it to use every available token value in [MIN, MAX]. "
                             "Not used for the Lx/Ly/eigenstate diagnostic axes.")
    parser.add_argument("--y-param", default=None, metavar="PARAM")
    parser.add_argument("--y-range", type=float, nargs="+", metavar="N", default=None,
                        help="MIN MAX [STEP]; see --x-range.")
    parser.add_argument("--n-occ", type=int, default=None,
                        help="Fixed particle number (default: half-filling)")


def _validate_range(name, rng):
    if rng is not None and len(rng) not in (2, 3):
        raise ValueError(f"--{name} takes 2 (MIN MAX) or 3 (MIN MAX STEP) values; got {len(rng)}.")


def _add_output_args(parser):
    parser.add_argument("--log-path", default=None, metavar="FILE",
                        help="JSON file to write the run log to")
    parser.add_argument("--plot-path", default=None, metavar="FILE",
                        help="PDF file to write the plot to")
    parser.add_argument("--hide-plot", dest="hide_plot", action="store_true", default=False)
    parser.add_argument("--hide-legend", action="store_true", default=False)


def _add_boundary_args(parser):
    parser.add_argument("--boundary", choices=["periodic", "hard-wall", "hard_wall", "open"],
                        default="periodic",
                        help="Real-space boundary condition for finite-lattice model runs (default: periodic).")


def _add_geometry_args(parser):
    parser.add_argument("--geometry", choices=["rectangle", "disk"], default="rectangle",
                        help="Finite real-space domain shape inside the parent lattice (default: rectangle).")
    parser.add_argument("--radius", type=float, default=None,
                        help="Disk radius; required when --geometry disk.")
    parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Disk center in real-space coordinates. Defaults to the parent lattice center.")


def _add_profile_args(parser):
    parser.add_argument("--potential-profile", choices=["none", "soft-dot", "soft_dot"], default="none",
                        help="Onsite scalar potential profile (default: none).")
    parser.add_argument("--potential-radius", type=float, default=None,
                        help="Radius for the soft-dot potential wall.")
    parser.add_argument("--potential-v0", type=float, default=None,
                        help="Outer potential height for the soft-dot profile.")
    parser.add_argument("--potential-xi", type=float, default=None,
                        help="Smoothing length for the soft-dot potential.")
    parser.add_argument("--mass-profile", choices=["none", "radial-step", "radial_step", "radial-tanh", "radial_tanh"], default="none",
                        help="Radial A/B Semenoff mass profile for Haldane-like models (default: none).")
    parser.add_argument("--mass-radius", type=float, default=None,
                        help="Radius for the radial mass interface.")
    parser.add_argument("--mass-inner", type=float, default=None,
                        help="Mass value inside the radial mass interface.")
    parser.add_argument("--mass-outer", type=float, default=None,
                        help="Mass value outside the radial mass interface.")
    parser.add_argument("--mass-xi", type=float, default=None,
                        help="Smoothing length for radial-tanh mass profiles.")
    parser.add_argument("--profile-center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Center for radial potential/mass profiles. Defaults to active geometry center.")


def _add_distribution_args(parser):
    parser.add_argument("--task-index", type=int, default=None, metavar="N")
    parser.add_argument("--task-count", type=int, default=1, metavar="N")
    parser.add_argument("--prepare-only", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--no-progress-log", action="store_true", default=False)


def _add_method_param_flags(parser):
    """Auto-generate --<method>-<param> flags from each method's PARAM_SPECS."""
    for m in METHOD_ORDER:
        cls = get_method_class(m)
        for spec in cls.PARAM_SPECS:
            if not spec.cli:
                continue
            flag = f"--{m.value}-{spec.name.replace('_', '-')}"
            dest = f"{m.value}_{spec.name}"
            if spec.is_flag:
                parser.add_argument(flag, dest=dest, default=None,
                                    action=argparse.BooleanOptionalAction, help=spec.help)
            else:
                parser.add_argument(flag, dest=dest, type=spec.type, default=None,
                                    metavar=spec.metavar or spec.name.upper(),
                                    choices=spec.choices, help=spec.help)


def _add_operator_dict_flags(parser):
    """Bespoke flags for the dict-valued VQE/IQPE operator-path parameters."""
    parser.add_argument("--vqe-ansatz", default=None, choices=list(_QISKIT_ANSATZES),
                        help="VQE ansatz type for the --qubit-operator path (default: efficient_su2).")
    parser.add_argument("--vqe-ansatz-kwarg", dest="vqe_ansatz_kwarg", action="append", default=None,
                        metavar="KEY=VALUE",
                        help="Ansatz kwarg; repeatable. Values may be numbers or @n_layers bindings.")
    parser.add_argument("--vqe-ansatz-prefix", dest="vqe_ansatz_prefix",
                        choices=["hartree_fock", "none"], default="none",
                        help="Initial-state prefix for the ansatz (default: none).")
    parser.add_argument("--vqe-optimizer", default=None, choices=list(_QISKIT_OPTIMIZERS),
                        help="Classical optimizer for the --qubit-operator path (default: SPSA).")
    parser.add_argument("--vqe-optimizer-kwarg", dest="vqe_optimizer_kwarg", action="append", default=None,
                        metavar="KEY=VALUE", help="Optimizer kwarg; repeatable, e.g. maxiter=@max_iters.")
    parser.add_argument("--iqpe-initial-state", dest="iqpe_initial_state", default=None,
                        choices=list(_INITIAL_STATES),
                        help="Initial state type for IQPE (default: uniform for operator path).")
    parser.add_argument("--iqpe-initial-vqe-ansatz", dest="iqpe_initial_vqe_ansatz", default=None,
                        choices=list(_QISKIT_ANSATZES),
                        help="VQE ansatz for vqe_informed initial state (default: efficient_su2).")
    parser.add_argument("--iqpe-initial-vqe-ansatz-kwarg", dest="iqpe_initial_vqe_ansatz_kwarg",
                        action="append", default=None, metavar="KEY=VALUE",
                        help="Ansatz kwarg for vqe_informed initial state; repeatable.")
    parser.add_argument("--iqpe-initial-vqe-n-layers", dest="iqpe_initial_vqe_n_layers", type=int,
                        default=None, metavar="N",
                        help="Ansatz layers for vqe_informed initial state (default: 1).")
    parser.add_argument("--iqpe-initial-vqe-max-iters", dest="iqpe_initial_vqe_max_iters", type=int,
                        default=None, metavar="N",
                        help="VQE iterations for vqe_informed initial state (default: 100).")


def _add_operator_args(parser):
    parser.add_argument("--qubit-operator", dest="qubit_operator", default=None,
                        nargs="+", metavar="SOURCE",
                        help="One or more HamLib HDF5 sources (local files, zip archives, or URLs).")
    parser.add_argument("--extremum", choices=["min", "max"], default="min",
                        help="Target eigenvalue for the --qubit-operator path (default: min).")
    parser.add_argument("--select", dest="select", action="append", default=None, metavar="TERMS",
                        help="Comma-separated key-segment filters to narrow the source to one "
                             "Hamiltonian family before sweeping (repeatable, AND semantics).")


def _parse_cli_kwargs(pairs):
    from qbp._console import _coerce_kwarg_value
    out = {}
    for item in (pairs or []):
        if "=" not in item:
            raise ValueError(f"kwarg '{item}' must be KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = _coerce_kwarg_value(v)
    return out


def _ansatz_dict(args):
    if getattr(args, "vqe_ansatz", None) is None:
        return None
    return {
        "type": args.vqe_ansatz,
        "kwargs": _parse_cli_kwargs(args.vqe_ansatz_kwarg),
        "initial_state_prefix": args.vqe_ansatz_prefix,
    }


def _optimizer_dict(args):
    if getattr(args, "vqe_optimizer", None) is None:
        return None
    return {"type": args.vqe_optimizer, "kwargs": _parse_cli_kwargs(args.vqe_optimizer_kwarg)}


def _iqpe_initial_state_dict(args):
    if getattr(args, "iqpe_initial_state", None) is None:
        return None
    d: dict = {"type": args.iqpe_initial_state}
    if args.iqpe_initial_state == "vqe_informed":
        if getattr(args, "iqpe_initial_vqe_ansatz", None) is not None:
            d["vqe_ansatz"] = {
                "type": args.iqpe_initial_vqe_ansatz,
                "kwargs": _parse_cli_kwargs(args.iqpe_initial_vqe_ansatz_kwarg),
                "initial_state_prefix": "hartree_fock",
            }
        if getattr(args, "iqpe_initial_vqe_n_layers", None) is not None:
            d["vqe_n_layers"] = args.iqpe_initial_vqe_n_layers
        if getattr(args, "iqpe_initial_vqe_max_iters", None) is not None:
            d["vqe_max_iters"] = args.iqpe_initial_vqe_max_iters
    return d


def _collect_method_params(args, methods):
    method_params: dict = {}
    for m in methods:
        cls = get_method_class(m)
        params: dict = {}
        for spec in cls.PARAM_SPECS:
            if not spec.cli:
                continue
            val = getattr(args, f"{m.value}_{spec.name}", None)
            if val is not None:
                params[spec.name] = val
        if m == Method.VQE:
            ansatz = _ansatz_dict(args)
            optimizer = _optimizer_dict(args)
            if ansatz is not None:
                params["ansatz"] = ansatz
            if optimizer is not None:
                params["optimizer"] = optimizer
        if m == Method.IQPE:
            initial = _iqpe_initial_state_dict(args)
            if initial is not None:
                params["initial_state"] = initial
        method_params[m] = params
    return method_params


def _resolve_backend(args):
    return getattr(args, "backend", None)


def _dispatch_run(args):
    methods = [Method.coerce(m) for m in args.method]
    method_params = _collect_method_params(args, methods)
    backend = _resolve_backend(args)

    if args.qubit_operator is not None:
        if args.model:
            raise ValueError("--model and --qubit-operator are mutually exclusive.")
        run(
            method=methods, method_params=method_params, backend=backend,
            qubit_operator=args.qubit_operator, extremum=args.extremum, select=args.select,
            x_param=args.x_param, x_range=args.x_range,
            y_param=args.y_param, y_range=args.y_range,
            log_path=args.log_path, plot_path=args.plot_path,
            hide_plot=args.hide_plot, hide_legend=args.hide_legend, heatmap=args.heatmap,
            diff=args.diff, diff_format=args.diff_format,
        )
        return

    if not args.model:
        raise ValueError("one of --model or --qubit-operator is required")
    model = get_model(args.model)
    run(
        model, method=methods, method_params=method_params, backend=backend,
        lattice=tuple(args.lattice) if args.lattice else None,
        boundary=args.boundary,
        geometry=args.geometry,
        radius=args.radius,
        center=args.center,
        potential_profile=args.potential_profile,
        potential_radius=args.potential_radius,
        potential_v0=args.potential_v0,
        potential_xi=args.potential_xi,
        mass_profile=args.mass_profile,
        mass_radius=args.mass_radius,
        mass_inner=args.mass_inner,
        mass_outer=args.mass_outer,
        mass_xi=args.mass_xi,
        profile_center=args.profile_center,
        x_param=args.x_param, x_range=args.x_range,
        y_param=args.y_param, y_range=args.y_range, n_occ=args.n_occ,
        model_params=_collect_model_params(args, model, args.x_param, args.y_param),
        observable=args.observable,
        log_path=args.log_path, plot_path=args.plot_path,
        hide_plot=args.hide_plot, hide_legend=args.hide_legend, heatmap=args.heatmap,
        diff=args.diff, diff_format=args.diff_format,
        task_index=args.task_index, task_count=args.task_count,
        prepare_only=args.prepare_only, aggregate_only=args.aggregate_only,
        no_progress_log=args.no_progress_log,
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        from qbp._console import run_console
        return run_console()

    if argv[0] == "register":
        if len(argv) == 1:
            from qbp._console import run_console
            return run_console(initial_command="register")
        if len(argv) == 3 and argv[1] == "--from":
            try:
                model = register_model_from_file(argv[2])
            except (FileNotFoundError, ValueError) as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"Registered '{model.name}'.")
            return 0
        print("usage: qbp register | qbp register --from <path.yaml>", file=sys.stderr)
        return 2

    if argv[0] == "remove":
        if len(argv) != 2:
            print("usage: qbp remove <model-name>", file=sys.stderr)
            return 2
        try:
            remove_model(argv[1])
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("command", nargs="?")
    pre.add_argument("--model", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(prog="qbp")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List available models or observables")
    list_parser.add_argument("target", choices=["models", "observables"])
    list_parser.add_argument("--model", default=None, metavar="MODEL",
                             help="Required when listing observables")

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("path", help="Path to a log JSON file")
    plot_parser.add_argument("--hide-plot", dest="hide_plot", action="store_true", default=False)
    plot_parser.add_argument("--output", default=None, metavar="PATH", help="Output PDF path")
    plot_parser.add_argument("--hide-legend", action="store_true", default=False)
    plot_parser.add_argument("--diff", action="store_true", default=False,
                             help="Also produce a difference plot for every pair of methods.")
    plot_parser.add_argument("--diff-format", dest="diff_format", default="3d",
                             choices=["3d", "heatmap", "bar_2d"],
                             help="Plot format for --diff plots (default: 3d).")

    run_parser = sub.add_parser("run", help="Run one or more simulation methods over a sweep")
    run_parser.add_argument("--model", default=None, metavar="MODEL",
                            help="Registered model name. Mutually exclusive with --qubit-operator.")
    run_parser.add_argument("--method", nargs="+", required=True,
                            choices=[m.value for m in METHOD_ORDER],
                            help="One or more simulation methods to run.")
    run_parser.add_argument("--lattice", type=int, nargs="+", default=None, metavar="N",
                            help="Lattice extents per dimension (omit for band-structure runs).")
    run_parser.add_argument("--observable", default="E", metavar="NAME",
                            help="Observable to compute per cell (default: 'E').")
    run_parser.add_argument("--heatmap", action="store_true", default=False,
                            help="Render as a 2D heatmap (one method + both sweep axes).")
    run_parser.add_argument("--diff", action="store_true", default=False,
                            help="Also produce a difference plot for every pair of methods.")
    run_parser.add_argument("--diff-format", dest="diff_format", default="3d",
                            choices=["3d", "heatmap", "bar_2d"],
                            help="Plot format for --diff plots (default: 3d).")
    run_parser.add_argument("--backend", default=None, metavar="NAME",
                            help="VQE/IQPE execution backend: a fake backend for local noise "
                                 "(e.g. FakeSherbrooke), a real IBM device (e.g. ibm_brisbane), "
                                 "or least_busy. Omit for ideal simulation.")
    _add_boundary_args(run_parser)
    _add_geometry_args(run_parser)
    _add_profile_args(run_parser)
    _add_sweep_args(run_parser)
    _add_output_args(run_parser)
    _add_distribution_args(run_parser)
    _add_operator_args(run_parser)
    _add_method_param_flags(run_parser)
    _add_operator_dict_flags(run_parser)

    if pre_args.command == "run" and pre_args.model:
        try:
            model = get_model(pre_args.model)
        except ValueError as e:
            parser.error(str(e))
        _add_model_params(run_parser, model)

    args = parser.parse_args(argv)

    try:
        _validate_range("x-range", getattr(args, "x_range", None))
        _validate_range("y-range", getattr(args, "y_range", None))
    except ValueError as e:
        parser.error(str(e))

    if args.command == "list":
        if args.target == "models":
            from qbp._registry import _MODELS
            if not _MODELS:
                print("(no models registered)")
            else:
                width = max(len(n) for n in _MODELS)
                for name in sorted(_MODELS):
                    print(f"  {name.ljust(width)}  {_MODELS[name].display_name}")
            return
        if args.target == "observables":
            if not args.model:
                parser.error("list observables requires --model NAME")
            try:
                model = get_model(args.model)
            except ValueError as e:
                parser.error(str(e))
            width = max(len(n) for n in model.observables)
            for name in sorted(model.observables):
                obs = model.observables[name]
                print(f"  {name.ljust(width)}  {obs.display_name}")
            return

    if args.command == "plot":
        try:
            result = load_result(args.path)
        except Exception as e:
            parser.error(str(e))
        result.plot(hide_plot=args.hide_plot, output_path=args.output, hide_legend=args.hide_legend,
                    diff=args.diff, diff_format=args.diff_format)
        return

    if args.command == "run":
        try:
            _dispatch_run(args)
        except (ValueError, FileNotFoundError, KeyError) as e:
            parser.error(str(e))
        return
