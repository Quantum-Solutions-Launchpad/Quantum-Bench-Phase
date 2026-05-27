from __future__ import annotations

import argparse
import sys

from quaph._registry import get_model, register_model_from_file, remove_model
from quaph._run import run_analytic, run_simulated_ideal, run_simulated_noisy
from quaph._yaml_model import _QISKIT_ANSATZES, _QISKIT_OPTIMIZERS


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
        y_param=args.y_param,
        y_range=args.y_range,
        ansatz=_ansatz_dict(args),
        optimizer=_optimizer_dict(args),
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

    for p in (sim_ideal_parser, sim_noisy_parser):
        _add_sim_required(p)
        _add_sim_optional(p)
        p.add_argument("--observable", default="E", metavar="NAME",
                       help="Observable to compute per cell (default: 'E'). "
                            "VQE supports any registered observable; IQPE supports only 'E' "
                            "and energy-based composites.")

    if subcommand in ("analytic", "simulated-ideal", "simulated-noisy") and model_arg:
        try:
            model = get_model(model_arg)
        except ValueError as e:
            parser.error(str(e))

        target = {
            "analytic": analytic_parser,
            "simulated-ideal": sim_ideal_parser,
            "simulated-noisy": sim_noisy_parser,
        }[subcommand]
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
        if args.subcommand == "analytic":
            _dispatch_analytic(args, model)
        elif args.subcommand == "simulated-ideal":
            _dispatch_simulated_ideal(args, model)
        elif args.subcommand == "simulated-noisy":
            _dispatch_simulated_noisy(args, model)
    except ValueError as e:
        parser.error(str(e))
