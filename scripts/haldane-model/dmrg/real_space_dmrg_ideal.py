from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DMRG Haldane sweep outputs and runtime summaries."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Path to the sweep JSON produced by scripts/julia-dmrg/dmrg_haldane.jl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots should be saved",
    )
    parser.add_argument(
        "--with-exact",
        action="store_true",
        help="Also generate per-run energy-vs-particle-number plots with exact overlays",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows",
    )
    return parser.parse_args()


def schedule_label(maxdim_schedule: list[int]) -> str:
    return "-".join(str(value) for value in maxdim_schedule)


def default_paths(root: Path) -> tuple[Path, Path]:
    input_json = root / "logs" / "haldane" / "dmrg" / "dmrg-sweep.json"
    output_dir = root / "plots" / "haldane" / "dmrg"
    return input_json, output_dir


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def mean_by_x(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for x_value, y_value in points:
        grouped[float(x_value)].append(float(y_value))

    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([np.mean(grouped[x]) for x in xs], dtype=float)
    return xs, ys


def aggregate_runs(runs: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        key = (
            int(run["n_sites"]),
            float(run["t2"]),
            tuple(int(value) for value in run["maxdim_schedule"]),
        )
        grouped[key].append(run)
    return grouped


def plot_runtime_vs_n_sites(runs: list[dict], output_dir: Path) -> Path | None:
    grouped_lines: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        key = (float(run["t2"]), schedule_label(run["maxdim_schedule"]))
        grouped_lines[key].append((run["n_sites"], run["summary"]["total_wall_time_s"]))

    if not grouped_lines:
        return None

    plt.figure(figsize=(8, 5))
    for (t2, maxdim_text), points in sorted(grouped_lines.items()):
        xs, ys = mean_by_x(points)
        plt.plot(xs, ys, marker="o", label=f"t2={t2:g}, maxdim={maxdim_text}")

    plt.xlabel("Number of sites")
    plt.ylabel("Mean total wall time (s)")
    plt.title("DMRG runtime scaling with system size")
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = output_dir / "runtime_vs_n_sites.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_runtime_vs_t2(runs: list[dict], output_dir: Path) -> Path | None:
    grouped_lines: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        key = (int(run["n_sites"]), schedule_label(run["maxdim_schedule"]))
        grouped_lines[key].append((run["t2"], run["summary"]["total_wall_time_s"]))

    if not grouped_lines:
        return None

    plt.figure(figsize=(8, 5))
    for (n_sites, maxdim_text), points in sorted(grouped_lines.items()):
        xs, ys = mean_by_x(points)
        plt.plot(xs, ys, marker="o", label=f"N={n_sites}, maxdim={maxdim_text}")

    plt.xlabel("$t_2$")
    plt.ylabel("Mean total wall time (s)")
    plt.title("DMRG runtime sensitivity to next-nearest-neighbor hopping")
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = output_dir / "runtime_vs_t2.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_dmrg_time_vs_n_occ(runs: list[dict], output_dir: Path) -> Path | None:
    grouped_runs = aggregate_runs(runs)
    if not grouped_runs:
        return None

    plt.figure(figsize=(9, 6))
    for key in sorted(grouped_runs):
        n_sites, t2, maxdim = key
        line_points: dict[int, list[float]] = defaultdict(list)
        for run in grouped_runs[key]:
            for sector in run["sectors"]:
                line_points[int(sector["n_occ"])].append(float(sector["profile"]["dmrg"]["elapsed_s"]))

        xs = np.array(sorted(line_points), dtype=int)
        ys = np.array([np.mean(line_points[x]) for x in xs], dtype=float)
        label = f"N={n_sites}, t2={t2:g}, maxdim={schedule_label(list(maxdim))}"
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Particle number ($n_{occ}$)")
    plt.ylabel("Mean DMRG time per sector (s)")
    plt.title("Per-sector DMRG runtime")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = output_dir / "dmrg_time_vs_n_occ.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_max_bond_vs_n_occ(runs: list[dict], output_dir: Path) -> Path | None:
    grouped_runs = aggregate_runs(runs)
    if not grouped_runs:
        return None

    plt.figure(figsize=(9, 6))
    for key in sorted(grouped_runs):
        n_sites, t2, maxdim = key
        line_points: dict[int, list[float]] = defaultdict(list)
        for run in grouped_runs[key]:
            for sector in run["sectors"]:
                line_points[int(sector["n_occ"])].append(float(sector["max_link_dim"]))

        xs = np.array(sorted(line_points), dtype=int)
        ys = np.array([np.mean(line_points[x]) for x in xs], dtype=float)
        label = f"N={n_sites}, t2={t2:g}, maxdim={schedule_label(list(maxdim))}"
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Particle number ($n_{occ}$)")
    plt.ylabel("Mean max bond dimension reached")
    plt.title("Observed bond dimension by particle-number sector")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = output_dir / "max_bond_vs_n_occ.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_energy_overlays(runs: list[dict], output_dir: Path) -> list[Path]:
    from utils import real_space_exact

    saved_paths: list[Path] = []
    grouped_runs = aggregate_runs(runs)

    for key in sorted(grouped_runs):
        n_sites, t2, maxdim = key
        run_group = grouped_runs[key]
        representative = run_group[0]
        n_occ_values = [int(value) for value in representative["n_occ_values"]]
        t1 = float(representative["t1"])
        phi = float(representative["phi"])

        dmrg_by_n_occ: dict[int, list[float]] = defaultdict(list)
        for run in run_group:
            for sector in run["sectors"]:
                dmrg_by_n_occ[int(sector["n_occ"])].append(float(sector["energy"]))

        mean_dmrg = np.array([np.mean(dmrg_by_n_occ[n_occ]) for n_occ in n_occ_values], dtype=float)
        exact = np.array(
            [real_space_exact(n_sites, t1, t2, phi, n_occ) for n_occ in n_occ_values],
            dtype=float,
        )

        plt.figure(figsize=(7, 4.5))
        plt.plot(n_occ_values, exact, "ro-", label="Exact")
        plt.plot(n_occ_values, mean_dmrg, "ko", label="DMRG")
        plt.xlabel("Particle number ($n_{occ}$)")
        plt.ylabel("$E$")
        plt.title(
            "DMRG vs exact energy\n"
            f"N={n_sites}, t2={t2:g}, maxdim={schedule_label(list(maxdim))}"
        )
        plt.legend()
        plt.tight_layout()

        file_path = output_dir / (
            f"energy_nsites-{n_sites}_t2-{t2:g}_maxdim-{schedule_label(list(maxdim))}.png"
        )
        plt.savefig(file_path, dpi=200)
        plt.close()
        saved_paths.append(file_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger = logging.getLogger(__name__)

    root = Path(__file__).resolve().parents[2]
    default_input, default_output = default_paths(root)
    input_json = args.input_json or default_input
    output_dir = args.output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(input_json)
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(
            f"{input_json} does not look like a sweep JSON with a top-level 'runs' list."
        )

    saved_paths: list[Path] = []
    for plotter in (
        plot_runtime_vs_n_sites,
        plot_runtime_vs_t2,
        plot_dmrg_time_vs_n_occ,
        plot_max_bond_vs_n_occ,
    ):
        path = plotter(runs, output_dir)
        if path is not None:
            saved_paths.append(path)

    if args.with_exact:
        saved_paths.extend(plot_energy_overlays(runs, output_dir))

    for path in saved_paths:
        logger.info("Saved plot: %s", path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
