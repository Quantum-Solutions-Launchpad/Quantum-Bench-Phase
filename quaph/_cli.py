from __future__ import annotations

import argparse
import sys

from quaph._registry import get_model, register_model_from_file, remove_model
from quaph._run import run_analytic, run_simulated_ideal, run_simulated_noisy
from quaph._realspace import plot_real_space_state_density
from quaph._edge import plot_edge_spectrum
from quaph._yaml_model import _QISKIT_ANSATZES, _QISKIT_OPTIMIZERS, _INITIAL_STATES


def _model_param_names(model) -> list[str]:
    momentum = set(model.momentum_axes)
    return [p for p in model.param_labels if p != "n_occ" and p not in momentum]


def _add_model_params(parser, model):
    for name in _model_param_names(model):
        parser.add_argument(
            f"--{name}", type=float, default=None, metavar=name.upper(),
            help=f"Value for model parameter {name} (required unless it is the active sweep axis)",
        )


def _resolve_sweep_axes(args) -> tuple[str | None, str | None]:
    return args.x_param, args.y_param


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
    parser.add_argument("--x-param", default=None, metavar="PARAM")
    parser.add_argument("--x-range", type=float, nargs="+", metavar="N", default=None,
                        help="MIN MAX [STEP]. Model sweeps require STEP; --qubit-operator "
                             "sweeps may omit it to use every available token value in [MIN, MAX].")
    parser.add_argument("--y-param", default=None, metavar="PARAM")
    parser.add_argument("--y-range", type=float, nargs="+", metavar="N", default=None,
                        help="MIN MAX [STEP]; see --x-range.")
    parser.add_argument("--n-occ", type=int, default=None,
                        help="Fixed particle number (default: half-filling)")


def _validate_range(name, rng):
    if rng is not None and len(rng) not in (2, 3):
        raise ValueError(f"--{name} takes 2 (MIN MAX) or 3 (MIN MAX STEP) values; got {len(rng)}.")


def _add_output_args(parser):
    parser.add_argument("--log-dir", default=None, metavar="PATH",
                        help="Directory for log JSON files (model/<lattice-tag>/ appended)")
    parser.add_argument("--plot-dir", default=None, metavar="PATH",
                        help="Directory for plot PDF files (model/<lattice-tag>/ appended)")
    parser.add_argument("--hide-plot", dest="hide_plot",
                        action="store_true", default=False)


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


def _add_sim_required(parser):
    parser.add_argument("--vqe-iters", type=int, default=None, metavar="N")
    parser.add_argument("--vqe-layers", type=int, default=None, metavar="N")
    parser.add_argument("--iqpe-time", type=float, default=None, metavar="F")
    parser.add_argument("--iqpe-trot", type=int, default=None, metavar="N")
    parser.add_argument("--iqpe-iters", type=int, default=None, metavar="N")


def _add_sim_optional(parser):
    parser.add_argument("--vqe-reps", type=int, default=None, metavar="N")
    parser.add_argument("--iqpe-reps", type=int, default=None, metavar="N")
    parser.add_argument("--hide-legend", action="store_true", default=False)


def _add_operator_args(parser, *, simulated):
    parser.add_argument("--qubit-operator", dest="qubit_operator", default=None, metavar="SOURCE",
                        help="HamLib HDF5 source: a local .h5/.hdf5 file, a local .zip archive "
                             "containing one, or an http(s) URL to either (e.g. a HamLib library "
                             ".zip link). Choose the sweep axes with --x-param/--y-param naming key "
                             "tokens (e.g. --x-param h --y-param Lx) and narrow multi-family sources "
                             "to one family with --select; with no axes it sweeps every key by "
                             "instance index.")
    parser.add_argument("--extremum", choices=["min", "max"], default="min",
                        help="Target eigenvalue for the --qubit-operator path (default: min).")
    parser.add_argument("--select", dest="select", action="append", default=None, metavar="TERMS",
                        help="Comma-separated key-segment filters to narrow the source to one "
                             "Hamiltonian family before sweeping (repeatable, AND semantics). "
                             "Terms match whole '-'/'_'-delimited segments, e.g. "
                             "--select 1D,grid,pbc or a token pin like --select Ly-105.")
    if simulated:
        parser.add_argument("--ansatz", default=None, choices=list(_QISKIT_ANSATZES),
                            help="VQE ansatz type for the --qubit-operator path (default: efficient_su2).")
        parser.add_argument("--ansatz-kwarg", dest="ansatz_kwarg", action="append", default=None,
                            metavar="KEY=VALUE",
                            help="Ansatz kwarg; repeatable. Values may be numbers or runtime bindings like @n_layers.")
        parser.add_argument("--ansatz-prefix", dest="ansatz_prefix", choices=["hartree_fock", "none"],
                            default="none", help="Initial-state prefix for the ansatz (default: none).")
        parser.add_argument("--optimizer", default=None, choices=list(_QISKIT_OPTIMIZERS),
                            help="Classical optimizer for the --qubit-operator path (default: SPSA).")
        parser.add_argument("--optimizer-kwarg", dest="optimizer_kwarg", action="append", default=None,
                            metavar="KEY=VALUE",
                            help="Optimizer kwarg; repeatable, e.g. maxiter=@max_iters.")
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


def _parse_cli_kwargs(pairs):
    from quaph._console import _coerce_kwarg_value
    out = {}
    for item in (pairs or []):
        if "=" not in item:
            raise ValueError(f"kwarg '{item}' must be KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = _coerce_kwarg_value(v)
    return out


def _ansatz_dict(args):
    if getattr(args, "ansatz", None) is None:
        return None
    return {
        "type": args.ansatz,
        "kwargs": _parse_cli_kwargs(args.ansatz_kwarg),
        "initial_state_prefix": args.ansatz_prefix,
    }


def _optimizer_dict(args):
    if getattr(args, "optimizer", None) is None:
        return None
    return {"type": args.optimizer, "kwargs": _parse_cli_kwargs(args.optimizer_kwarg)}


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


def _dispatch_analytic_operator(args):
    run_analytic(
        qubit_operator=args.qubit_operator,
        extremum=args.extremum,
        select=args.select,
        x_param=args.x_param,
        x_range=args.x_range,
        y_param=args.y_param,
        y_range=args.y_range,
        heatmap=args.heatmap,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        hide_plot=args.hide_plot,
    )


def _dispatch_simulated_operator(run_fn, args):
    run_fn(
        qubit_operator=args.qubit_operator,
        extremum=args.extremum,
        select=args.select,
        x_param=args.x_param,
        x_range=args.x_range,
        y_param=y_param,
        y_range=args.y_range,
        ansatz=_ansatz_dict(args),
        optimizer=_optimizer_dict(args),
        iqpe_initial_state=_iqpe_initial_state_dict(args),
        vqe_iters=args.vqe_iters,
        vqe_layers=args.vqe_layers,
        vqe_reps=args.vqe_reps,
        iqpe_time=args.iqpe_time,
        iqpe_trot=args.iqpe_trot,
        iqpe_iters=args.iqpe_iters,
        iqpe_reps=args.iqpe_reps,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        hide_plot=args.hide_plot,
        hide_legend=args.hide_legend,
    )


def _dispatch_analytic(args, model):
    x_param, y_param = _resolve_sweep_axes(args)
    run_analytic(
        model,
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
        x_param=x_param,
        x_range=args.x_range,
        y_param=y_param,
        y_range=args.y_range,
        n_occ=args.n_occ,
        model_params=_collect_model_params(args, model, x_param, y_param),
        observable=args.observable,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        hide_plot=args.hide_plot,
        heatmap=args.heatmap,
    )


def _dispatch_simulated_ideal(args, model):
    x_param, y_param = _resolve_sweep_axes(args)
    run_simulated_ideal(
        model,
        lattice=tuple(args.lattice) if args.lattice else None,
        boundary=args.boundary,
        x_param=x_param,
        x_range=args.x_range,
        y_param=y_param,
        y_range=args.y_range,
        n_occ=args.n_occ,
        model_params=_collect_model_params(args, model, x_param, y_param),
        vqe_iters=args.vqe_iters,
        vqe_layers=args.vqe_layers,
        vqe_reps=args.vqe_reps,
        iqpe_time=args.iqpe_time,
        iqpe_trot=args.iqpe_trot,
        iqpe_iters=args.iqpe_iters,
        iqpe_reps=args.iqpe_reps,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        hide_plot=args.hide_plot,
        hide_legend=args.hide_legend,
    )


def _dispatch_simulated_noisy(args, model):
    x_param, y_param = _resolve_sweep_axes(args)
    run_simulated_noisy(
        model,
        lattice=tuple(args.lattice) if args.lattice else None,
        boundary=args.boundary,
        x_param=x_param,
        x_range=args.x_range,
        y_param=y_param,
        y_range=args.y_range,
        n_occ=args.n_occ,
        model_params=_collect_model_params(args, model, x_param, y_param),
        vqe_iters=args.vqe_iters,
        vqe_layers=args.vqe_layers,
        vqe_reps=args.vqe_reps,
        iqpe_time=args.iqpe_time,
        iqpe_trot=args.iqpe_trot,
        iqpe_iters=args.iqpe_iters,
        iqpe_reps=args.iqpe_reps,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        hide_plot=args.hide_plot,
        hide_legend=args.hide_legend,
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        from quaph._console import run_console
        return run_console()

    if argv and argv[0] == "register":
        if len(argv) == 1:
            from quaph._console import run_console
            return run_console(initial_command="register")
        if len(argv) == 3 and argv[1] == "--from":
            try:
                model = register_model_from_file(argv[2])
            except (FileNotFoundError, ValueError) as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"Registered '{model.name}'.")
            return 0
        print("usage: quaph register | quaph register --from <path.yaml>", file=sys.stderr)
        return 2

    if argv and argv[0] == "remove":
        if len(argv) != 2:
            print("usage: quaph remove <model-name>", file=sys.stderr)
            return 2
        try:
            remove_model(argv[1])
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("command", nargs="?")
    pre.add_argument("subcommand", nargs="?")
    pre.add_argument("--model", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    subcommand = pre_args.subcommand
    model_arg = pre_args.model

    parser = argparse.ArgumentParser(prog="quaph")
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

    plot_state_parser = sub.add_parser("plot-state", help="Plot a real-space eigenstate density")
    plot_state_parser.add_argument("--model", default=None, metavar="MODEL",
                                   help="Registered model name (e.g. haldane, hubbard, haldane-hubbard).")
    plot_state_parser.add_argument("--lattice", type=int, nargs="+", required=True, metavar="N",
                                   help="Lattice extents per dimension (e.g. --lattice 3 3).")
    plot_state_parser.add_argument("--boundary", choices=["periodic", "hard-wall", "hard_wall", "open"],
                                   default="periodic",
                                   help="Real-space boundary condition for the finite-lattice model (default: periodic).")
    plot_state_parser.add_argument("--geometry", choices=["rectangle", "disk"], default="rectangle",
                                   help="Finite real-space domain shape inside the parent lattice (default: rectangle).")
    plot_state_parser.add_argument("--radius", type=float, default=None,
                                   help="Disk radius; required when --geometry disk.")
    plot_state_parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                                   help="Disk center in real-space coordinates. Defaults to the parent lattice center.")
    plot_state_parser.add_argument("--state-index", type=int, default=None,
                                   help="Eigenstate index after sorting energies ascending. Defaults to the state closest to E=0.")
    plot_state_parser.add_argument("--n-occ", type=int, default=None,
                                   help="Select the highest occupied state for this filling, i.e. state index n_occ - 1.")
    plot_state_parser.add_argument("--view", choices=["2d", "3d"], default="2d",
                                   help="Density view to render (default: 2d).")
    plot_state_parser.add_argument("--output", default=None, metavar="PATH",
                                   help="Output image/PDF path. Omit to show an interactive window.")
    plot_state_parser.add_argument("--hide-plot", dest="hide_plot", action="store_true", default=False)
    plot_state_parser.add_argument("--no-bonds", dest="show_bonds", action="store_false", default=True,
                                   help="Hide hopping bonds in the real-space plot.")
    plot_state_parser.add_argument("--max-bonds", type=int, default=3000,
                                   help="Maximum number of bonds to draw (default: 3000).")
    _add_profile_args(plot_state_parser)

    edge_spectrum_parser = sub.add_parser(
        "edge-spectrum",
        help="Plot eigenenergies colored by edge participation",
    )
    edge_spectrum_parser.add_argument("--model", default=None, metavar="MODEL",
                                      help="Registered model name (e.g. haldane, hubbard, haldane-hubbard).")
    edge_spectrum_parser.add_argument("--lattice", type=int, nargs="+", required=True, metavar="N",
                                      help="Lattice extents per dimension (e.g. --lattice 10 10).")
    edge_spectrum_parser.add_argument("--boundary", choices=["periodic", "hard-wall", "hard_wall", "open"],
                                      default="hard-wall",
                                      help="Real-space boundary condition for the finite-lattice model (default: hard-wall).")
    edge_spectrum_parser.add_argument("--geometry", choices=["rectangle", "disk"], default="rectangle",
                                      help="Finite real-space domain shape inside the parent lattice (default: rectangle).")
    edge_spectrum_parser.add_argument("--radius", type=float, default=None,
                                      help="Disk radius; required when --geometry disk.")
    edge_spectrum_parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                                      help="Disk center in real-space coordinates. Defaults to the parent lattice center.")
    edge_spectrum_parser.add_argument("--output", default=None, metavar="PATH",
                                      help="Output PDF/image path. Omit to show an interactive window.")
    edge_spectrum_parser.add_argument("--hide-plot", dest="hide_plot", action="store_true", default=False)
    _add_profile_args(edge_spectrum_parser)

    run_parser = sub.add_parser("run")
    run_sub = run_parser.add_subparsers(dest="subcommand", required=True)

    analytic_parser = run_sub.add_parser("analytic")
    sim_ideal_parser = run_sub.add_parser("simulated-ideal")
    sim_noisy_parser = run_sub.add_parser("simulated-noisy")

    for p in (analytic_parser, sim_ideal_parser, sim_noisy_parser):
        p.add_argument("--model", default=None, metavar="MODEL",
                       help="Registered model name (e.g. haldane, hubbard, haldane-hubbard). "
                            "Mutually exclusive with --qubit-operator.")
        p.add_argument("--lattice", type=int, nargs="+", default=None, metavar="N",
                       help="Lattice extents per dimension (e.g. --lattice 3 3 for a 3x3 unit-cell grid). Omit for momentum-space band-structure runs.")
        p.add_argument("--boundary", choices=["periodic", "hard-wall", "hard_wall", "open"], default="periodic",
                       help="Real-space boundary condition for finite-lattice model runs (default: periodic).")
        _add_sweep_args(p)
        _add_output_args(p)

    _add_operator_args(analytic_parser, simulated=False)
    for p in (sim_ideal_parser, sim_noisy_parser):
        _add_operator_args(p, simulated=True)

    analytic_parser.add_argument("--heatmap", action="store_true", default=False,
                                 help="Render results as a 2D heatmap (requires both x and y sweep axes)")
    analytic_parser.add_argument("--observable", default="E", metavar="NAME",
                                 help="Observable to compute per cell (default: 'E'). "
                                      "Use 'list observables --model NAME' to see what's available.")
    analytic_parser.add_argument("--geometry", choices=["rectangle", "disk"], default="rectangle",
                                 help="Finite real-space domain shape inside the parent lattice (default: rectangle).")
    analytic_parser.add_argument("--radius", type=float, default=None,
                                 help="Disk radius; required when --geometry disk.")
    analytic_parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                                 help="Disk center in real-space coordinates. Defaults to the parent lattice center.")
    _add_profile_args(analytic_parser)

    for p in (sim_ideal_parser, sim_noisy_parser):
        _add_sim_required(p)
        _add_sim_optional(p)
        p.add_argument("--observable", default="E", metavar="NAME",
                       help="Observable to compute per cell (default: 'E'). "
                            "VQE supports any registered observable; IQPE supports only 'E' "
                            "and energy-based composites.")

    if (
        (subcommand in ("analytic", "simulated-ideal", "simulated-noisy"))
        or pre_args.command in ("plot-state", "edge-spectrum")
    ) and model_arg:
        try:
            model = get_model(model_arg)
        except ValueError as e:
            parser.error(str(e))

        target = (
            {
                "plot-state": plot_state_parser,
                "edge-spectrum": edge_spectrum_parser,
            }.get(pre_args.command)
            or {
                "analytic": analytic_parser,
                "simulated-ideal": sim_ideal_parser,
                "simulated-noisy": sim_noisy_parser,
            }[subcommand]
        )
        _add_model_params(target, model)

    args = parser.parse_args(argv)

    try:
        _validate_range("x-range", getattr(args, "x_range", None))
        _validate_range("y-range", getattr(args, "y_range", None))
    except ValueError as e:
        parser.error(str(e))

    if args.command == "list":
        if args.target == "models":
            from quaph._registry import _MODELS
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
        from quaph._run import load_result
        try:
            result = load_result(args.path)
        except Exception as e:
            parser.error(str(e))
        kwargs = dict(hide_plot=args.hide_plot, output_path=args.output)
        if hasattr(result, "vqe_best_energies"):
            kwargs["hide_legend"] = args.hide_legend
        result.plot(**kwargs)
        return

    if getattr(args, "qubit_operator", None) is not None:
        if args.model:
            parser.error("--model and --qubit-operator are mutually exclusive.")
        try:
            if args.subcommand == "analytic":
                _dispatch_analytic_operator(args)
            elif args.subcommand == "simulated-ideal":
                _dispatch_simulated_operator(run_simulated_ideal, args)
            elif args.subcommand == "simulated-noisy":
                _dispatch_simulated_operator(run_simulated_noisy, args)
        except (ValueError, FileNotFoundError, KeyError) as e:
            parser.error(str(e))
        return

    if not args.model:
        parser.error("one of --model or --qubit-operator is required")

    try:
        model = get_model(args.model)
    except ValueError as e:
        parser.error(str(e))

    try:
        if args.command == "plot-state":
            params = _collect_model_params(args, model, None, None)
            plot_real_space_state_density(
                model=model,
                lattice=args.lattice,
                model_params=params,
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
                state_index=args.state_index,
                n_occ=args.n_occ,
                view=args.view,
                show_bonds=args.show_bonds,
                max_bonds=args.max_bonds,
                output_path=args.output,
                hide_plot=args.hide_plot,
            )
        elif args.command == "edge-spectrum":
            params = _collect_model_params(args, model, None, None)
            plot_edge_spectrum(
                model=model,
                lattice=args.lattice,
                model_params=params,
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
                output_path=args.output,
                hide_plot=args.hide_plot,
            )
        elif args.subcommand == "analytic":
            _dispatch_analytic(args, model)
        elif args.subcommand == "simulated-ideal":
            _dispatch_simulated_ideal(args, model)
        elif args.subcommand == "simulated-noisy":
            _dispatch_simulated_noisy(args, model)
    except ValueError as e:
        parser.error(str(e))
