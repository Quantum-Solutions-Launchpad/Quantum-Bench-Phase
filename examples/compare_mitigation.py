"""General raw-vs-mitigated(-vs-mitigated...) comparison

Runs the same parameter grid under any number of named mitigation configs
against a shared noise model, plots each run's error surface against the
analytic answer, and prints a numerical summary comparing them all.

This is a separate, more capable tool alongside the existing run_diff.py
(which only plots a single pre-existing log — no run, no comparison).

Usage: edit the CONFIGS section at the bottom, or import
compare_strategies() and call it from your own script:

    /opt/homebrew/bin/python3.11 examples/compare_mitigation.py
"""
import os
import json
import numpy as np
import qbp
from qbp import Method


def _grid_errors(raw_data_path, method):
    """Error of every individual repetition against analytic, pooled across
    the whole grid — reads the raw-data dump (not the single reduced value
    per cell) to avoid a cherry-picking bias when comparing configs. See
    run_mitigation_sweep.py's matching function for the full rationale.
    """
    with open(raw_data_path) as f:
        data = json.load(f)
    cells = data["cells"]
    analytic = cells.get("analytic", {})
    sim = cells.get(method, {})
    errors = []
    for xi_str, row in sim.items():
        for yi_str, cell in row.items():
            a_cell = analytic.get(xi_str, {}).get(yi_str)
            if a_cell is None or cell is None:
                continue
            a_val = a_cell.get("value")
            if a_val is None and "bands" in a_cell:
                a_val = min(a_cell["bands"])
            if a_val is None:
                continue
            for r in cell.get("repetitions") or []:
                errors.append(abs(float(r) - float(a_val)))
    return np.array(errors)


def _rmse(errs):
    return float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("nan")


def compare_strategies(
    *, model, lattice, x_param, x_range, y_param, y_range, model_params,
    method, method_params, backend, configs, out_dir, tag,
    plot_format="heatmap",
):
    """Run the same grid under each (name, mitigation_dict) in *configs*
    against the shared *backend* noise model, plot each run's error, and
    print a numerical summary comparing all configs against the first
    (the baseline — usually raw/no mitigation).

    Parameters
    ----------
    method : Method
        Method.VQE or Method.IQPE.
    method_params : dict
        Base parameters for *method* (e.g. iters/layers/reps for VQE,
        time/trot/iters/reps for IQPE) WITHOUT the "mitigation" key —
        that's merged in per-config from *configs*.
    configs : list[tuple[str, dict]]
        e.g. [("raw", {}), ("zne", {"zne": True}), ("dd", {"dd": True})].
        Each dict becomes that run's "mitigation" config.
    out_dir : str
        Directory for logs/plots (created if missing).
    tag : str
        Short name used in output filenames (e.g. "zne-vs-dd").

    Returns
    -------
    dict[str, np.ndarray]
        Per-config arrays of |repetition - analytic| across the whole grid.
    """
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    shared = dict(
        model=model, lattice=lattice,
        x_param=x_param, x_range=x_range,
        y_param=y_param, y_range=y_range,
        model_params=model_params, hide_plot=True,
    )
    method_key = method.value

    results = {}
    for name, mitigation in configs:
        log_path = os.path.join(out_dir, "logs", f"{tag}-{name}.json")
        params = {**method_params, "mitigation": mitigation}

        if not os.path.exists(log_path):
            print(f"Running '{name}'...")
            qbp.run(**shared, method=[Method.ANALYTIC, method],
                      method_params={method: params},
                      backend=backend, log_path=log_path)

        else:
            print(f"'{name}' log exists, skipping.")

        diff_path = os.path.join(out_dir, "plots", f"{tag}-{name}-diff-{plot_format}.pdf")
        qbp.plot_diff(log_path, method=method_key, plot_format=plot_format,
                         output_path=diff_path)


        raw_data_path = log_path.replace(".json", ".raw-data.json")
        results[name] = _grid_errors(raw_data_path, method_key)

    # Numerical summary
    baseline_name = configs[0][0]
    baseline_errors = results[baseline_name]
    print("\n" + "=" * 70)
    print(f"COMPARISON — {tag}  (baseline: '{baseline_name}')")
    print("=" * 70)
    header = f"  {'Metric':<23}" + "".join(f"{name:>14}" for name, _ in configs)
    print(header)
    for label, fn in [
        ("Mean abs error (MAE)", lambda e: e.mean()),
        ("Median abs error", lambda e: np.median(e)),
        ("RMSE", _rmse),
        ("Max abs error", lambda e: e.max()),
    ]:
        row = f"  {label:<23}"
        for name, _ in configs:
            row += f"{fn(results[name]):>14.4f}"
        print(row)
    print(f"  {'Grid points':<23}" + "".join(f"{len(results[name]):>14}" for name, _ in configs))

    print(f"\n  MAE change vs. '{baseline_name}':")
    for name, _ in configs[1:]:
        r, m = baseline_errors.mean(), results[name].mean()
        pct = 100 * (r - m) / r if r > 0 else float("nan")
        note = ""
        if len(results[name]) == len(baseline_errors):
            helps = int((results[name] < baseline_errors).sum())
            note = f"   (helps {helps}/{len(baseline_errors)} points — index-paired, not cell-matched)"
        print(f"    {name:<20} {pct:+.1f}%{note}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import math
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    _HERE = os.path.dirname(os.path.abspath(__file__))

    def my_noise():
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 2), ["ecr", "cx"])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["sx", "x"])
        return nm

    compare_strategies(
        model="haldane-honeycomb", lattice=(2, 2),
        x_param="n_occ", x_range=(2, 6, 2),
        y_param="t2", y_range=(0.2, 0.8, 0.3),
        model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
        method=Method.VQE,
        method_params={"iters": 3000, "layers": 4, "reps": 4},
        backend=AerSimulator(noise_model=my_noise()),
        configs=[
            ("raw", {}),
            ("zne", {"zne": True}),
        ],
        out_dir=os.path.join(_HERE, "compare_output"),
        tag="vqe-gate-noise",
    )