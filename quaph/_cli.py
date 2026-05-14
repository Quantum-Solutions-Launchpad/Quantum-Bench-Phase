from __future__ import annotations

import argparse
import sys

from quaph._registry import get_model, remove_model
from quaph._run import run_analytic, run_simulated_ideal, run_simulated_noisy


def _model_param_names(model) -> list[str]:
    return [p for p in model.param_labels if p != "n_occ"]


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
    parser.add_argument("--x-range", type=float, nargs=3,
                        metavar=("MIN", "MAX", "STEP"), default=None)
    parser.add_argument("--y-param", default=None, metavar="PARAM")
    parser.add_argument("--y-range", type=float, nargs=3,
                        metavar=("MIN", "MAX", "STEP"), default=None)
    parser.add_argument("--n-occ", type=int, default=None,
                        help="Fixed particle number (default: half-filling)")


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

    if argv == ["register"]:
        from quaph._console import run_console
        return run_console(initial_command="register")

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
        p.add_argument("--model", required=True, metavar="MODEL",
                       help="Registered model name (e.g. haldane, hubbard, haldane-hubbard)")
        p.add_argument("--lattice", type=int, nargs="+", default=None, metavar="N",
                       help="Lattice extents per dimension (e.g. --lattice 3 3 for a 3x3 unit-cell grid). Omit for momentum-space band-structure runs.")
        _add_sweep_args(p)
        _add_output_args(p)

    analytic_parser.add_argument("--heatmap", action="store_true", default=False,
                                 help="Render results as a 2D heatmap (requires both x and y sweep axes)")

    for p in (sim_ideal_parser, sim_noisy_parser):
        _add_sim_required(p)
        _add_sim_optional(p)

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
