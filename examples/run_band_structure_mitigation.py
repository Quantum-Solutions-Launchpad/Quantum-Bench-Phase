"""Band structure IQPE: raw vs M3-mitigated comparison.

Uses IQPE (not VQE) because:
- IQPE M3 works correctly locally via mthree on bitstring counts
- VQE M3 via EstimatorV2 options is silently ignored in local mode
- IQPE naturally uses Z-basis bitstring measurement so mthree correction is exact

Noise model: amplified readout error (8%) isolates M3's target noise source.
"""
import os
import math
import json
import numpy as np
import quaph
from quaph import Method
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

MODEL = "haldane"
X_PARAM = "kx"
Y_PARAM = "ky"
STEP = math.pi / 4 
MODEL_PARAMS = {"t1": 1.0, "t2": 0.05, "M": 0.2, "phi": math.pi / 2}

_HERE = os.path.dirname(os.path.abspath(__file__))

def _readout_noise_model(p_error: float = 0.08):
    """Pure amplified readout error, M3's target noise regime.

    Build from scratch (not NoiseModel.from_backend) so the injected
    all-qubit error isn't silently blocked by pre-existing per-qubit
    calibrated readout errors on the fake backend.
    """
    nm = NoiseModel()
    ro_err = ReadoutError([[1 - p_error, p_error], [p_error, 1 - p_error]])
    nm.add_all_qubit_readout_error(ro_err)
    return nm

np.random.seed(42)
BACKEND = AerSimulator(noise_model=_readout_noise_model())

SHARED = dict(
    model=MODEL,
    x_param=X_PARAM,
    x_range=(-math.pi, math.pi, STEP),
    y_param=Y_PARAM,
    y_range=(-math.pi, math.pi, STEP),
    model_params=MODEL_PARAMS,
    hide_plot=True,
)

# time=0.15: gives phase~0.07 for E~-3, safely in [0,1)
# trot=8: reduces Trotter error to ~1% at this time step
# iters=6: gives resolution 1/64~0.016 in phase space
IQPE_PARAMS = {"time": 0.15, "trot": 8, "iters": 6, "reps": 3}

raw_log  = os.path.join(_HERE, "logs", "haldane", "band-iqpe-raw.json")
m3_log   = os.path.join(_HERE, "logs", "haldane", "band-iqpe-m3.json")
raw_plot = os.path.join(_HERE, "plots", "haldane", "band-iqpe-raw.pdf")
m3_plot  = os.path.join(_HERE, "plots", "haldane", "band-iqpe-m3.pdf")

# --- Raw IQPE ---
if os.path.exists(raw_log):
    print("Raw log exists — skipping raw IQPE run.")
else:
    print("Running raw IQPE band structure...")
    quaph.run(
        **SHARED,
        method=[Method.ANALYTIC, Method.IQPE],
        method_params={Method.IQPE: IQPE_PARAMS},
        backend=BACKEND,
        log_path=raw_log,
        plot_path=raw_plot,
    )

# --- M3-mitigated IQPE ---
if os.path.exists(m3_log):
    print("M3 log exists — skipping M3 IQPE run.")
else:
    print("Running M3-mitigated IQPE band structure...")
    quaph.run(
        **SHARED,
        method=[Method.ANALYTIC, Method.IQPE],
        method_params={Method.IQPE: {**IQPE_PARAMS, "mitigation": {"m3": True}}},
        backend=BACKEND,
        log_path=m3_log,
        plot_path=m3_plot,
    )

# --- Diff plots ---
print("Generating diff plots...")
quaph.plot_diff(
    raw_log, method="iqpe", plot_format="heatmap",
    output_path=os.path.join(_HERE, "plots", "haldane", "band-iqpe-raw-diff-heatmap.pdf"),
)
quaph.plot_diff(
    m3_log, method="iqpe", plot_format="heatmap",
    output_path=os.path.join(_HERE, "plots", "haldane", "band-iqpe-m3-diff-heatmap.pdf"),
)
quaph.plot_diff(
    raw_log, method="iqpe", plot_format="3d",
    output_path=os.path.join(_HERE, "plots", "haldane", "band-iqpe-raw-diff-3d.pdf"),
)
quaph.plot_diff(
    m3_log, method="iqpe", plot_format="3d",
    output_path=os.path.join(_HERE, "plots", "haldane", "band-iqpe-m3-diff-3d.pdf"),
)

# --- Numerical summary ---
def _grid_errors(log_path, method="iqpe"):
    """Error of every individual repetition against analytic
    """
    raw_data_path = log_path.replace(".json", ".raw-data.json")
    with open(raw_data_path) as f:
        data = json.load(f)
    cells = data["cells"]
    analytic = cells.get("analytic", {})
    sim      = cells.get(method, {})
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

raw_errs = _grid_errors(raw_log)
m3_errs  = _grid_errors(m3_log)

print("\n" + "="*57)
print("NUMERICAL SUMMARY — IQPE error vs analytic across k-space")
print("="*57)
print(f"{'Metric':<25} {'Raw IQPE':>10} {'M3 IQPE':>10} {'Improvement':>10}")
print("-"*57)
for label, r, m in [
    ("Mean abs error (MAE)", raw_errs.mean(), m3_errs.mean()),
    ("Median abs error",     np.median(raw_errs), np.median(m3_errs)),
    ("RMSE",                 _rmse(raw_errs), _rmse(m3_errs)),
    ("Max abs error",        raw_errs.max(), m3_errs.max()),
]:
    pct = 100 * (r - m) / r if r > 0 else float("nan")
    print(f"  {label:<23} {r:>10.4f} {m:>10.4f} {pct:>+9.1f}%")
print("-"*57)
print(f"  Grid points:           {len(raw_errs)}")
print(f"  Points M3 helps:       {(m3_errs < raw_errs).sum()} / {len(raw_errs)}")
print(f"  Points M3 hurts:       {(m3_errs > raw_errs).sum()} / {len(raw_errs)}")

print("\nDone. Outputs in examples/plots/haldane/:")
print("  band-iqpe-raw.pdf              — raw IQPE band structure")
print("  band-iqpe-m3.pdf               — M3 IQPE band structure")
print("  band-iqpe-raw-diff-heatmap.pdf — raw error heatmap over k-space")
print("  band-iqpe-m3-diff-heatmap.pdf  — M3 error heatmap over k-space")