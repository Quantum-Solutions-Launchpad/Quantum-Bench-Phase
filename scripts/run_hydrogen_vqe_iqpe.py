#!/usr/bin/env python3
"""
VQE + IQPE sweep over hydrogen systems.
Runs simulations for n=2,4,6,8,10,12 and R=0.5,0.8,1.1,1.4,1.7,2.0
using simulated ideal backend.
"""

import os
import json
from pathlib import Path
from itertools import product

import qbp
from qbp import Method

REPO_ROOT = os.environ.get("REPO_ROOT", "/pscratch/sd/m/mbao202/NNL-P7")
HAMLIB_BASE = os.environ.get(
    "HAMLIB_BASE",
    "https://portal.nersc.gov/cfs/m888/dcamps/hamlib_v1.0/chemistry"
)

# Parameters
N_VALUES = [2, 4, 6, 8, 10, 12]
R_VALUES = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]

# Output directories
OUTPUT_ROOT = Path(REPO_ROOT) / "manuscript-plots"
LOG_DIR = OUTPUT_ROOT / "logs" / "new-data" / "hydrogen" / "H-linear" / "ground-state-energy"
PLOT_DIR = OUTPUT_ROOT / "plots" / "new-data" / "hydrogen" / "H-linear" / "ground-state-energy"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# VQE parameters (same as haldane run)
VQE_ITERS = int(os.environ.get("VQE_ITERS", "10000"))
VQE_LAYERS = int(os.environ.get("VQE_LAYERS", "5"))
VQE_REPS = int(os.environ.get("VQE_REPS", "10"))

# IQPE parameters (same as haldane run)
IQPE_TIME = float(os.environ.get("IQPE_TIME", "0.2"))
IQPE_TROT = int(os.environ.get("IQPE_TROT", "5"))
IQPE_ITERS = int(os.environ.get("IQPE_ITERS", "8"))
IQPE_REPS = int(os.environ.get("IQPE_REPS", "20"))
IQPE_INITIAL_VQE_N_LAYERS = int(os.environ.get("IQPE_INITIAL_VQE_N_LAYERS", "2"))
IQPE_INITIAL_VQE_REPS = int(os.environ.get("IQPE_INITIAL_VQE_REPS", "2"))
IQPE_INITIAL_VQE_MAX_ITERS = int(os.environ.get("IQPE_INITIAL_VQE_MAX_ITERS", "1000"))

# Hamlib path
HAMLIB_PATH = os.environ.get("HAMLIB_PATH", f"{HAMLIB_BASE}/hydrogen_linear/H_linear.zip")


def format_r(r):
    """Format R value for hamlib queries (0.5 -> R0.5)"""
    return f"R{r}"


def format_n(n):
    """Format n value for hamlib queries (2 -> n2 or H2)"""
    return f"H{n}"


def run_hydrogen_sweep():
    """Run VQE + IQPE over hydrogen parameter space."""

    print("=" * 80)
    print("Hydrogen Linear: Ground-State Energy VQE + IQPE Sweep")
    print("=" * 80)
    print(f"n values: {N_VALUES}")
    print(f"R values: {R_VALUES}")
    print(f"Hamlib path: {HAMLIB_PATH}")
    print(f"Output directory: {LOG_DIR}")
    print("=" * 80)

    results = []

    for n, r in product(N_VALUES, R_VALUES):
        run_tag = f"H{n}_R{r}"
        log_path = LOG_DIR / f"H{n}_R{r}_vqe_iqpe.json"
        plot_path = PLOT_DIR / f"H{n}_R{r}_vqe_iqpe.pdf"

        print(f"\nRunning: {run_tag}")
        print(f"  Log: {log_path}")
        print(f"  Plot: {plot_path}")

        try:
            qbp.run(
                method=[Method.VQE, Method.IQPE],
                qubit_operator=HAMLIB_PATH,
                observable="E",
                select=["H-linear", format_n(n), format_r(r)],
                method_params={
                    Method.VQE: {
                        "iters": VQE_ITERS,
                        "layers": VQE_LAYERS,
                        "reps": VQE_REPS,
                    },
                    Method.IQPE: {
                        "time": IQPE_TIME,
                        "trot": IQPE_TROT,
                        "iters": IQPE_ITERS,
                        "reps": IQPE_REPS,
                        "initial_state": "vqe_informed",
                        "initial_vqe_ansatz": "excitation_preserving",
                        "initial_vqe_n_layers": IQPE_INITIAL_VQE_N_LAYERS,
                        "initial_vqe_ansatz_kwarg": f"reps={IQPE_INITIAL_VQE_REPS}",
                        "initial_vqe_max_iters": IQPE_INITIAL_VQE_MAX_ITERS,
                    },
                },
                log_path=str(log_path),
                plot_path=str(plot_path),
                hide_plot=True,
            )

            results.append({
                "n": n,
                "R": r,
                "status": "completed",
                "log": str(log_path),
                "plot": str(plot_path),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "n": n,
                "R": r,
                "status": "failed",
                "error": str(e),
            })

    # Save summary
    summary_path = LOG_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Sweep completed. Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_hydrogen_sweep()
