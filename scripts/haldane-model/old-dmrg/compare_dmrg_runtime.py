import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_laptop_runs(path: Path):
    runs = json.loads(path.read_text())["runs"]
    aggregated = {}
    for run in runs:
        key = (
            int(run["n_sites"]),
            float(run["t2"]),
            tuple(run["maxdim_schedule"]),
            int(run["nsweeps"]),
            float(run["cutoff"]),
            bool(run["conserve_qns"]),
            tuple(run["n_occ_values"]),
        )
        summary = run["summary"]
        entry = aggregated.setdefault(
            key,
            {
                "laptop_seeds": [],
                "laptop_total_wall_time_s": 0.0,
                "laptop_dmrg_wall_time_s": 0.0,
                "laptop_hamiltonian_build_wall_time_s": 0.0,
                "laptop_energies": [],
            },
        )
        entry["laptop_seeds"].append(int(run["seed"]))
        entry["laptop_total_wall_time_s"] += float(summary["total_wall_time_s"])
        entry["laptop_dmrg_wall_time_s"] += float(summary["dmrg_wall_time_s"])
        entry["laptop_hamiltonian_build_wall_time_s"] += float(
            summary["hamiltonian_build_wall_time_s"]
        )
        entry["laptop_energies"].extend(float(energy) for energy in run["energies"])
    return aggregated


def load_perlmutter_runs(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        key = (
            int(row["n_sites"]),
            float(row["t2"]),
            tuple(row["maxdim"]),
            int(row["nsweeps"]),
            float(row["cutoff"]),
            bool(row["conserve_qns"]),
            tuple(row["n_occ_values"]),
        )
        indexed[key] = row
    return indexed


def build_comparison_df(laptop_runs, perlmutter_runs):
    records = []
    for key in sorted(set(laptop_runs) & set(perlmutter_runs)):
        n_sites, t2, maxdim, nsweeps, cutoff, conserve_qns, n_occ_values = key
        laptop = laptop_runs[key]
        perlmutter = perlmutter_runs[key]
        summary = perlmutter["summary"]

        laptop_energies = sorted(round(energy, 10) for energy in laptop["laptop_energies"])
        perlmutter_energies = sorted(round(float(energy), 10) for energy in perlmutter["energies"])
        energy_match = laptop_energies == perlmutter_energies
        max_abs_energy_diff = (
            max(abs(a - b) for a, b in zip(laptop_energies, perlmutter_energies))
            if len(laptop_energies) == len(perlmutter_energies)
            else None
        )

        perlmutter_total = float(summary["total_wall_time_s"])
        perlmutter_dmrg = float(summary["dmrg_wall_time_s"])
        perlmutter_hamiltonian = float(summary["hamiltonian_build_wall_time_s"])

        records.append(
            {
                "n_sites": n_sites,
                "t2": t2,
                "maxdim_schedule": ",".join(str(x) for x in maxdim),
                "nsweeps": nsweeps,
                "cutoff": cutoff,
                "conserve_qns": conserve_qns,
                "num_sectors": len(n_occ_values),
                "laptop_seeds": ",".join(str(seed) for seed in sorted(laptop["laptop_seeds"])),
                "perlmutter_seeds": ",".join(str(seed) for seed in perlmutter["seeds"]),
                "laptop_total_wall_time_s": laptop["laptop_total_wall_time_s"],
                "perlmutter_total_wall_time_s": perlmutter_total,
                "total_speedup_perlmutter_vs_laptop": laptop["laptop_total_wall_time_s"]
                / perlmutter_total,
                "laptop_dmrg_wall_time_s": laptop["laptop_dmrg_wall_time_s"],
                "perlmutter_dmrg_wall_time_s": perlmutter_dmrg,
                "dmrg_speedup_perlmutter_vs_laptop": laptop["laptop_dmrg_wall_time_s"]
                / perlmutter_dmrg,
                "laptop_hamiltonian_build_wall_time_s": laptop[
                    "laptop_hamiltonian_build_wall_time_s"
                ],
                "perlmutter_hamiltonian_build_wall_time_s": perlmutter_hamiltonian,
                "hamiltonian_speedup_perlmutter_vs_laptop": laptop[
                    "laptop_hamiltonian_build_wall_time_s"
                ]
                / perlmutter_hamiltonian,
                "energies_match": energy_match,
                "max_abs_energy_diff": max_abs_energy_diff,
            }
        )

    return pd.DataFrame.from_records(records)


def make_plot(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, t2 in zip(axes, sorted(df["t2"].unique())):
        subset = df[df["t2"] == t2].sort_values("n_sites")
        ax.plot(
            subset["n_sites"],
            subset["laptop_total_wall_time_s"],
            marker="o",
            linewidth=2,
            label="Laptop",
        )
        ax.plot(
            subset["n_sites"],
            subset["perlmutter_total_wall_time_s"],
            marker="o",
            linewidth=2,
            label="Perlmutter",
        )
        ax.set_title(f"t2 = {t2:g}")
        ax.set_xlabel("n_sites")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Total wall time (s)")
    axes[0].legend(frameon=False)
    fig.suptitle("DMRG Runtime: Laptop vs Perlmutter")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laptop-json", required=True)
    parser.add_argument("--perlmutter-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plot", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    laptop_runs = load_laptop_runs(Path(args.laptop_json))
    perlmutter_runs = load_perlmutter_runs(Path(args.perlmutter_jsonl))
    df = build_comparison_df(laptop_runs, perlmutter_runs)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    make_plot(df, Path(args.output_plot))

    matched = len(df)
    print(f"Wrote {matched} matched comparisons to {output_csv}")
    print(f"Wrote plot to {args.output_plot}")


if __name__ == "__main__":
    main()
