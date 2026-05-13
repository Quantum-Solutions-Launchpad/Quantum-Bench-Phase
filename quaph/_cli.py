from __future__ import annotations

import argparse
import sys

from quaph._registry import get_model, remove_model
from quaph._run import run_analytic, run_simulated_ideal, run_simulated_noisy


def _sweep_param_names(model) -> set[str]:
    names = set()
    for ax_def in model.sweep_defaults.values():
        p = ax_def.get("param")
        if p and p != "n_occ" and p not in model.default_params:
            names.add(p)
    return names


def _add_model_params(parser, model):
    for name, val in model.default_params.items():
        parser.add_argument(f"--{name}", type=type(val), default=val,
                            metavar=name.upper())
    for name in sorted(_sweep_param_names(model)):
        parser.add_argument(f"--{name}", type=float, default=None,
                            metavar=name.upper(),
                            help=f"Fix {name} to a value (only valid when {name} is not the active sweep axis)")


def _resolve_sweep_axes(args, model) -> tuple[str, str]:
    """Return (x_param, y_param) as the library will resolve them."""
    sd = model.sweep_defaults
    x_param = args.x_param or sd.get("x", {"param": "n_occ"}).get("param", "n_occ")
    y_param = args.y_param or sd.get("y", {"param": "n_occ"}).get("param", "n_occ")
    return x_param, y_param


def _collect_model_params(args, model, x_param: str, y_param: str) -> dict:
    params = {k: getattr(args, k) for k in model.default_params}
    for name in _sweep_param_names(model):
        val = getattr(args, name, None)
        if val is not None:
            if name == x_param or name == y_param:
                raise ValueError(
                    f"--{name} cannot be used as a fixed value while '{name}' is the active "
                    f"sweep axis. Override the sweep with --x-param/--y-param first."
                )
            params[name] = val
    return params


def _add_sweep_args(parser):
    parser.add_argument("--x-param", default=None, metavar="PARAM")
    parser.add_argument("--x-range", type=float, nargs=3,
                        metavar=("MIN", "MAX", "STEP"), default=None)
    parser.add_argument("--y-param", default=None, metavar="PARAM")
    parser.add_argument("--y-range", type=float, nargs=3,
                        metavar=("MIN", "MAX", "STEP"), default=None)
    parser.add_argument("--n-occ", type=int, default=None,
                        help="Fixed particle number (default: n_sites)")


def _add_output_args(parser):
    parser.add_argument("--log-dir", default=None, metavar="PATH",
                        help="Directory for log JSON files (model/n-sites/ appended)")
    parser.add_argument("--plot-dir", default=None, metavar="PATH",
                        help="Directory for plot PDF files (model/n-sites/ appended)")
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
    x_param, y_param = _resolve_sweep_axes(args, model)
    run_analytic(
        model,
        n_sites=args.n_sites,
        x_param=args.x_param,
        x_range=args.x_range,
        y_param=args.y_param,
        y_range=args.y_range,
        n_occ=args.n_occ,
        model_params=_collect_model_params(args, model, x_param, y_param),
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        hide_plot=args.hide_plot,
    )


def _dispatch_simulated_ideal(args, model):
    x_param, y_param = _resolve_sweep_axes(args, model)
    run_simulated_ideal(
        model,
        n_sites=args.n_sites,
        x_param=args.x_param,
        x_range=args.x_range,
        y_param=args.y_param,
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
    x_param, y_param = _resolve_sweep_axes(args, model)
    run_simulated_noisy(
        model,
        n_sites=args.n_sites,
        x_param=args.x_param,
        x_range=args.x_range,
        y_param=args.y_param,
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
        p.add_argument("--n-sites", type=int, required=True, metavar="N")
        _add_sweep_args(p)
        _add_output_args(p)

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
        from quaph._run import load_result, SimulatedResult
        try:
            result = load_result(args.path)
        except Exception as e:
            parser.error(str(e))
        kwargs = dict(hide_plot=args.hide_plot, output_path=args.output)
        if isinstance(result, SimulatedResult):
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
