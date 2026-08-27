"""Compare raw vs DD-mitigated VQE across a small (n_occ, t2) grid.

Produces two logs (raw and DD) plus diff-of-diffs visualisations showing
where in parameter space dynamical decoupling helps most
"""
import math
import os
import json
import numpy as np
import quaph
from quaph import Method
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

_HERE = os.path.dirname(os.path.abspath(__file__))


def _decoherence_only_noise_model(t1_us: float = 15.0, t2_us: float = 10.0):
    nm = NoiseModel()
    t1, t2 = t1_us * 1e-6, t2_us * 1e-6
    for gate, gate_time in [("sx", 50e-9), ("x", 50e-9)]:
        err = thermal_relaxation_error(t1, t2, gate_time)
        nm.add_all_qubit_quantum_error(err, gate)
    err_2q_time = 300e-9
    err_q0 = thermal_relaxation_error(t1, t2, err_2q_time)
    err_q1 = thermal_relaxation_error(t1, t2, err_2q_time)
    nm.add_all_qubit_quantum_error(err_q0.tensor(err_q1), ["ecr", "cx"])
    return nm


BACKEND = AerSimulator(noise_model=_decoherence_only_noise_model())
MODEL_PARAMS = {"t1": 1.0, "phi": math.pi / 4, "M": 0.1}

SHARED = dict(
    model="haldane-honeycomb",
    lattice=(2, 2),
    x_param="n_occ",
    x_range=(2, 6, 2),       # n_occ = 2, 4, 6
    y_param="t2",
    y_range=(0.2, 0.8, 0.3),  # t2 = 0.2, 0.5, 0.8
    model_params=MODEL_PARAMS,
    hide_plot=True,
)

VQE_PARAMS = {"iters": 1200, "layers": 6, "reps": 6}

print("Running analytic reference...")
quaph.run(
    **SHARED, method=[Method.ANALYTIC], backend=None,
    log_path=os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-analytic.json"),
)

print("Running raw VQE sweep (no mitigation)...")
quaph.run(
    **SHARED,
    method=[Method.ANALYTIC, Method.VQE],
    method_params={"vqe": VQE_PARAMS},
    backend=BACKEND,
    log_path=os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-raw.json"),
)

print("Running DD-mitigated VQE sweep...")
quaph.run(
    **SHARED,
    method=[Method.ANALYTIC, Method.VQE],
    method_params={"vqe": {**VQE_PARAMS, "mitigation": {"dd": True}}},
    backend=BACKEND,
    log_path=os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-dd.json"),
)

print("\nGenerating diff plots...")
quaph.plot_diff(
    os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-raw.json"),
    method="vqe", plot_format="heatmap",
    output_path=os.path.join(_HERE, "plots/haldane/2x2/dd-sweep-raw-diff.pdf"),
)
quaph.plot_diff(
    os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-dd.json"),
    method="vqe", plot_format="heatmap",
    output_path=os.path.join(_HERE, "plots/haldane/2x2/dd-sweep-dd-diff.pdf"),
)
quaph.plot_diff(
    os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-raw.json"),
    method="vqe", plot_format="3d",
    output_path=os.path.join(_HERE, "plots/haldane/2x2/dd-sweep-raw-diff-3d.pdf"),
)
quaph.plot_diff(
    os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-dd.json"),
    method="vqe", plot_format="3d",
    output_path=os.path.join(_HERE, "plots/haldane/2x2/dd-sweep-dd-diff-3d.pdf"),
)

print("\nDone. Plots saved to examples/plots/haldane/2x2/")
print("  dd-sweep-raw-diff.pdf / dd-sweep-dd-diff.pdf (heatmaps)")
print("  dd-sweep-raw-diff-3d.pdf / dd-sweep-dd-diff-3d.pdf (3D surfaces)")


# -----------------------------------------------------------------------
# Numerical summary
# -----------------------------------------------------------------------
def _grid_errors(raw_data_path, method="vqe"):
    """Error of every individual repetition against analytic
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
    return float(np.sqrt(np.mean(errs ** 2)))


raw_errs = _grid_errors(os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-raw.raw-data.json"))
dd_errs = _grid_errors(os.path.join(_HERE, "logs/haldane/2x2/dd-sweep-dd.raw-data.json"))

print("\n" + "=" * 65)
print("NUMERICAL SUMMARY — VQE + DD error vs analytic across n_occ x t2")
print("=" * 65)
print(f"  {'Metric':<23} {'Raw':>10} {'DD':>10} {'Improvement':>12}")
for label, r, m in [
    ("Mean abs error (MAE)", raw_errs.mean(), dd_errs.mean()),
    ("Median abs error", np.median(raw_errs), np.median(dd_errs)),
    ("RMSE", _rmse(raw_errs), _rmse(dd_errs)),
    ("Max abs error", raw_errs.max(), dd_errs.max()),
]:
    pct = 100 * (r - m) / r if r > 0 else float("nan")
    print(f"  {label:<23} {r:>10.4f} {m:>10.4f} {pct:>+11.1f}%")
helps = int((dd_errs < raw_errs).sum())
hurts = int((dd_errs > raw_errs).sum())
print(f"  Grid points:            {len(raw_errs)}")
print(f"  Points DD helps:        {helps} / {len(raw_errs)}")
print(f"  Points DD hurts:        {hurts} / {len(raw_errs)}")
print("=" * 65)