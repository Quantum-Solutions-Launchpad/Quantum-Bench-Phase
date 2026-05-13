from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 3D Haldane DMRG phase diagram from merged JSONL output."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to the merged JSONL produced from per-rank DMRG outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots should be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows.",
    )
    return parser.parse_args()


def infer_n_sites_step(records: list[dict]) -> int:
    n_sites_values = sorted({int(record["n_sites"]) for record in records})
    if len(n_sites_values) < 2:
        return 1

    steps = [curr - prev for prev, curr in zip(n_sites_values, n_sites_values[1:])]
    positive_steps = [step for step in steps if step > 0]
    return min(positive_steps) if positive_steps else 1


def default_output_dir(root: Path, records: list[dict]) -> Path:
    max_sites = max(int(record["n_sites"]) for record in records)
    step = infer_n_sites_step(records)
    folder_name = f"phase-diagram-3d_n_sites{max_sites}_step{step}"
    return root / "plots" / "haldane" / "dmrg" / folder_name


def finalize_figure(fig: plt.Figure, path: Path) -> None:
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.06, top=0.88)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            if "error" in payload:
                logging.warning("Skipping error record on line %d: %s", lineno, payload["error"])
                continue
            records.append(payload)
    return records


def collect_all_energy_points(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    colors: list[float] = []

    for record in records:
        n_sites = float(record["n_sites"])
        t2 = float(record["t2"])
        for sector in record.get("sectors", []):
            xs.append(n_sites)
            ys.append(t2)
            zs.append(float(sector["energy"]))
            colors.append(float(sector["n_occ"]))

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(colors, dtype=float),
    )


def collect_ground_state_surface(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: dict[tuple[int, float], float] = {}
    for record in records:
        key = (int(record["n_sites"]), float(record["t2"]))
        energies = [float(value) for value in record.get("energies", [])]
        if not energies:
            continue
        energy_min = min(energies)
        if key not in points or energy_min < points[key]:
            points[key] = energy_min

    xs = np.asarray([key[0] for key in points], dtype=float)
    ys = np.asarray([key[1] for key in points], dtype=float)
    zs = np.asarray([points[key] for key in points], dtype=float)
    return xs, ys, zs


def describe_parameters(records: list[dict]) -> str:
    representative = records[0]
    n_sites_values = sorted({int(record["n_sites"]) for record in records})
    t2_values = sorted({float(record["t2"]) for record in records})
    t1 = float(representative["t1"])

    return (
        f"n_sites={n_sites_values[0]}-{n_sites_values[-1]}, "
        f"t2={t2_values[0]:g}-{t2_values[-1]:g}, "
        f"t1={t1:g}"
    )


def plot_all_energy_points(records: list[dict], output_dir: Path, parameter_text: str) -> Path | None:
    xs, ys, zs, colors = collect_all_energy_points(records)
    if xs.size == 0:
        return None

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=colors, cmap="viridis", s=18, alpha=0.7)
    ax.set_xlabel("n_sites")
    ax.set_ylabel("$t_2$")
    ax.set_zlabel("Energy")
    ax.set_title(f"Haldane DMRG phase diagram: all sector energies\n{parameter_text}")
    fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.8, label="$n_{occ}$")

    path = output_dir / "energy_phase_diagram_3d_all_points.png"
    finalize_figure(fig, path)
    return path


def plot_ground_state_surface(records: list[dict], output_dir: Path, parameter_text: str) -> Path | None:
    xs, ys, zs = collect_ground_state_surface(records)
    if xs.size == 0:
        return None

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    trisurf = ax.plot_trisurf(xs, ys, zs, cmap="plasma", edgecolor="none", alpha=0.9)
    ax.scatter(xs, ys, zs, color="black", s=16, alpha=0.85)
    ax.set_xlabel("n_sites")
    ax.set_ylabel("$t_2$")
    ax.set_zlabel("Min energy")
    ax.set_title(f"Haldane DMRG phase diagram: minimum energy surface\n{parameter_text}")
    fig.colorbar(trisurf, ax=ax, pad=0.12, shrink=0.8, label="Min energy")

    path = output_dir / "energy_phase_diagram_3d_ground_state.png"
    finalize_figure(fig, path)
    return path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger = logging.getLogger(__name__)

    root = Path(__file__).resolve().parents[2]
    records = load_jsonl(args.input_jsonl)
    if not records:
        raise ValueError(f"No valid records found in {args.input_jsonl}.")
    output_dir = args.output_dir or default_output_dir(root, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_text = describe_parameters(records)

    saved_paths: list[Path] = []
    for plotter in (
        plot_all_energy_points,
        plot_ground_state_surface,
    ):
        path = plotter(records, output_dir, parameter_text)
        if path is not None:
            saved_paths.append(path)

    for path in saved_paths:
        logger.info("Saved plot: %s", path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
