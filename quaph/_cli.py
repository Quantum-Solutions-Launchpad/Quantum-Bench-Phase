from __future__ import annotations

import argparse
import os
import sys

from quaph._registry import get_model, register_model_from_file, remove_model
from quaph._run import run_analytic, run_simulated_ideal, run_simulated_noisy
from quaph._dmrg import run_dmrg_itensor


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
    parser.add_argument("--task-index", type=int, default=None, metavar="N")
    parser.add_argument("--task-count", type=int, default=1, metavar="N")
    parser.add_argument("--prepare-only", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--no-progress-log", action="store_true", default=False)


def _add_dmrg_args(parser):
    parser.add_argument("--julia", default="julia", metavar="PATH",
                        help="Julia executable to use for ITensorMPS DMRG")
    parser.add_argument("--julia-module", default="julia/1.11.7", metavar="NAME",
                        help="Environment module to load if --julia is not on PATH")
    parser.add_argument("--julia-project", default=None, metavar="PATH",
                        help="Julia project containing ITensors, ITensorMPS, and JSON")
    parser.add_argument("--dmrg-script", default=None, metavar="PATH",
                        help="Override the bundled Julia DMRG bridge script")
    parser.add_argument("--nsweeps", type=int, default=4, metavar="N")
    parser.add_argument("--maxdims", default="20,50,100,200", metavar="LIST")
    parser.add_argument("--cutoff", type=float, default=1e-9, metavar="F")
    parser.add_argument("--seed", type=int, default=1234, metavar="N")
    parser.add_argument("--conserve-qns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-index", type=int, default=None, metavar="N")
    parser.add_argument("--task-count", type=int, default=1, metavar="N")
    parser.add_argument("--prepare-only", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--no-progress-log", action="store_true", default=False)


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
        task_index=args.task_index,
        task_count=args.task_count,
        prepare_only=args.prepare_only,
        aggregate_only=args.aggregate_only,
        no_progress_log=args.no_progress_log,
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
        task_index=args.task_index,
        task_count=args.task_count,
        prepare_only=args.prepare_only,
        aggregate_only=args.aggregate_only,
        no_progress_log=args.no_progress_log,
    )


def _dispatch_dmrg(args, model):
    x_param, y_param = _resolve_sweep_axes(args)
    result = run_dmrg_itensor(
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
        julia=args.julia,
        julia_module=args.julia_module,
        julia_project=args.julia_project,
        nsweeps=args.nsweeps,
        maxdims=args.maxdims,
        cutoff=args.cutoff,
        seed=args.seed,
        conserve_qns=args.conserve_qns,
        script_path=args.dmrg_script,
        task_index=args.task_index,
        task_count=args.task_count,
        prepare_only=args.prepare_only,
        aggregate_only=args.aggregate_only,
        no_progress_log=args.no_progress_log,
    )
    if "summary_path" in result:
        print(f"Wrote {result['summary_path']}")
    elif result.get("type") == "dmrg-shard":
        print(
            f"Completed DMRG shard {result['task_index']}/{result['task_count']} "
            f"({result['num_cells']} cells)"
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

    if len(argv) >= 2 and argv[0] == "run" and argv[1] in (
        "real-space-simulated-ideal",
        "real-space-simulated-noisy",
    ):
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from _real_space_simulated_common import main as run_real_space_simulated

        simulation_tag = "ideal" if argv[1] == "real-space-simulated-ideal" else "noisy"
        return run_real_space_simulated(simulation_tag, argv=argv[2:])

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
    dmrg_parser = run_sub.add_parser("dmrg")

    for p in (analytic_parser, sim_ideal_parser, sim_noisy_parser, dmrg_parser):
        p.add_argument("--model", required=True, metavar="MODEL",
                       help="Registered model name (e.g. haldane, hubbard, haldane-hubbard)")
        p.add_argument("--lattice", type=int, nargs="+", default=None, metavar="N",
                       help="Lattice extents per dimension (e.g. --lattice 3 3 for a 3x3 unit-cell grid). Omit for momentum-space band-structure runs.")
        _add_sweep_args(p)
        _add_output_args(p)

    analytic_parser.add_argument("--heatmap", action="store_true", default=False,
                                 help="Render results as a 2D heatmap (requires both x and y sweep axes)")
    analytic_parser.add_argument("--observable", default="E", metavar="NAME",
                                 help="Observable to compute per cell (default: 'E'). "
                                      "Use 'list observables --model NAME' to see what's available.")

    for p in (sim_ideal_parser, sim_noisy_parser):
        _add_sim_required(p)
        _add_sim_optional(p)
    _add_dmrg_args(dmrg_parser)

    if subcommand in ("analytic", "simulated-ideal", "simulated-noisy", "dmrg") and model_arg:
        try:
            model = get_model(model_arg)
        except ValueError as e:
            parser.error(str(e))

        target = {
            "analytic": analytic_parser,
            "simulated-ideal": sim_ideal_parser,
            "simulated-noisy": sim_noisy_parser,
            "dmrg": dmrg_parser,
        }[subcommand]
        _add_model_params(target, model)

    args = parser.parse_args(argv)

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
        elif args.subcommand == "dmrg":
            _dispatch_dmrg(args, model)
    except ValueError as e:
        parser.error(str(e))
