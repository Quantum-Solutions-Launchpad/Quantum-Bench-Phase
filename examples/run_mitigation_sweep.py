#Mitigation testing of raw vs mitigated VQE and IQPE across n_occ x t2 grid

#Three noise regimes where we try to the find technique that works best in each:
#- VQE + ZNE:  pure gate depolarizing error
#- VQE + DD:   pure decoherence/T1-T2
#- IQPE + M3:  amplified readout error

#Each produces a pair of diff heatmaps (raw vs mitigated error surface)

import os
import math
import json
import numpy as np
import warnings
import qbp
from qbp import Method
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
)
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

warnings.filterwarnings("ignore")

MODEL    = "haldane-honeycomb"
LATTICE  = (2, 2)
X_PARAM  = "n_occ"
Y_PARAM  = "t2"
MODEL_PARAMS = {"t1": 1.0, "phi": math.pi / 4, "M": 0.1}

_HERE = os.path.dirname(os.path.abspath(__file__))

# Noise models

def _gate_error_only(p_2q=0.08, p_1q=0.01):
    # 2q error registered on both "cx" and "ecr": the transpiler's chosen
    # 2-qubit gate depends on the backend's basis_gates and can vary, and
    # NoiseModel.add_all_qubit_quantum_error only ever fires for the exact
    # gate NAME registered.
    nm = NoiseModel()
    err_2q = depolarizing_error(p_2q, 2)
    nm.add_all_qubit_quantum_error(err_2q, ["ecr", "cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), ["sx", "x"])
    return nm

def _decoherence_only(t1_us=15.0, t2_us=10.0):
    nm = NoiseModel()
    t1, t2 = t1_us * 1e-6, t2_us * 1e-6
    for gate, gt in [("sx", 50e-9), ("x", 50e-9)]:
        nm.add_all_qubit_quantum_error(thermal_relaxation_error(t1, t2, gt), gate)
    err_q0 = thermal_relaxation_error(t1, t2, 300e-9)
    err_q1 = thermal_relaxation_error(t1, t2, 300e-9)
    err_2q = err_q0.tensor(err_q1)
    nm.add_all_qubit_quantum_error(err_2q, ["ecr", "cx"])
    return nm

def _readout_only(p_error=0.08):
    """Pure readout error only — no gate errors, no decoherence.
    Build from scratch so per-qubit errors don't block all-qubit override.
    """
    nm = NoiseModel()
    ro_err = ReadoutError([[1 - p_error, p_error], [p_error, 1 - p_error]])
    nm.add_all_qubit_readout_error(ro_err)
    return nm

BACKEND_ZNE = AerSimulator(noise_model=_gate_error_only(p_2q=0.002, p_1q=0.0005))
BACKEND_DD  = AerSimulator(noise_model=_decoherence_only(t1_us=150.0, t2_us=100.0))
BACKEND_M3  = AerSimulator(noise_model=_readout_only())

SHARED = dict(
    model=MODEL, lattice=LATTICE,
    x_param=X_PARAM, x_range=(2, 6, 2),   # n_occ = 2, 4, 6
    y_param=Y_PARAM, y_range=(0.2, 0.8, 0.3),  # t2 = 0.2, 0.5, 0.8
    model_params=MODEL_PARAMS,
    hide_plot=True,
)

def log(name): return os.path.join(_HERE, "logs",  "haldane", "2x2", f"sweep-{name}.json")
def plt(name): return os.path.join(_HERE, "plots", "haldane", "2x2", f"sweep-{name}.pdf")

os.makedirs(os.path.join(_HERE, "logs",  "haldane", "2x2"), exist_ok=True)
os.makedirs(os.path.join(_HERE, "plots", "haldane", "2x2"), exist_ok=True)

# Experiment A: VQE + ZNE vs raw (gate error)

print("\n=== A: VQE + ZNE (gate depolarizing noise) ===")

VQE_ZNE = {"iters": 3000, "layers": 4, "reps": 4}

if not os.path.exists(log("vqe-raw-gate")):
    print("Running raw VQE...")
    qbp.run(**SHARED, method=[Method.ANALYTIC, Method.VQE],
              method_params={Method.VQE: VQE_ZNE},
              backend=BACKEND_ZNE,
              log_path=log("vqe-raw-gate"), plot_path=plt("vqe-raw-gate"))
else:
    print("Raw VQE log exists, skipping.")

if not os.path.exists(log("vqe-zne-gate")):
    print("Running ZNE VQE...")
    qbp.run(**SHARED, method=[Method.ANALYTIC, Method.VQE],
              method_params={Method.VQE: {**VQE_ZNE, "mitigation": {"zne": True}}},
              backend=BACKEND_ZNE,
              log_path=log("vqe-zne-gate"), plot_path=plt("vqe-zne-gate"))
else:
    print("ZNE VQE log exists, skipping.")

# Experiment B: VQE + DD vs raw (decoherence)

print("\n=== B: VQE + DD (decoherence noise) ===")

VQE_DD = {"iters": 3000, "layers": 6, "reps": 6}

if not os.path.exists(log("vqe-raw-decoherence")):
    print("Running raw VQE...")
    qbp.run(**SHARED, method=[Method.ANALYTIC, Method.VQE],
              method_params={Method.VQE: VQE_DD},
              backend=BACKEND_DD,
              log_path=log("vqe-raw-decoherence"), plot_path=plt("vqe-raw-decoherence"))
else:
    print("Raw VQE log exists, skipping.")

if not os.path.exists(log("vqe-dd-decoherence")):
    print("Running DD VQE...")
    qbp.run(**SHARED, method=[Method.ANALYTIC, Method.VQE],
              method_params={Method.VQE: {**VQE_DD, "mitigation": {"dd": True}}},
              backend=BACKEND_DD,
              log_path=log("vqe-dd-decoherence"), plot_path=plt("vqe-dd-decoherence"))
else:
    print("DD VQE log exists, skipping.")

# Experiment C: IQPE + M3 vs raw (readout error)

print("\n=== C: IQPE + M3 (readout error) ===")

IQPE_M3 = {
    "time": 0.5, "trot": 4, "iters": 6, "reps": 6,
    "warm_start_vqe": True, "warm_start_iters": 3000, "warm_start_layers": 4,
}

SHARED_IQPE_LO = {**SHARED, "x_range": (2, 3, 1)}  # n_occ = 2, 3
SHARED_IQPE_HI = {**SHARED, "x_range": (5, 6, 1)}  # n_occ = 5, 6

for suffix, grid in [("-lo", SHARED_IQPE_LO), ("-hi", SHARED_IQPE_HI)]:
    if not os.path.exists(log(f"iqpe-raw-readout{suffix}")):
        print(f"Running raw IQPE ({suffix})...")
        qbp.run(**grid, method=[Method.ANALYTIC, Method.IQPE],
                  method_params={Method.IQPE: IQPE_M3},
                  backend=BACKEND_M3,
                  log_path=log(f"iqpe-raw-readout{suffix}"), plot_path=plt(f"iqpe-raw-readout{suffix}"))
    else:
        print(f"Raw IQPE log exists, skipping ({suffix}).")

    if not os.path.exists(log(f"iqpe-m3-readout{suffix}")):
        print(f"Running M3 IQPE ({suffix})...")
        qbp.run(**grid, method=[Method.ANALYTIC, Method.IQPE],
                  method_params={Method.IQPE: {**IQPE_M3, "mitigation": {"m3": True}}},
                  backend=BACKEND_M3,
                  log_path=log(f"iqpe-m3-readout{suffix}"), plot_path=plt(f"iqpe-m3-readout{suffix}"))
    else:
        print(f"M3 IQPE log exists, skipping ({suffix}).")


# Diff plots

print("\nGenerating diff plots...")
for name, method in [
    ("vqe-raw-gate",        "vqe"),
    ("vqe-zne-gate",        "vqe"),
    ("vqe-raw-decoherence", "vqe"),
    ("vqe-dd-decoherence",  "vqe"),
    ("iqpe-raw-readout-lo", "iqpe"),
    ("iqpe-raw-readout-hi", "iqpe"),
    ("iqpe-m3-readout-lo",  "iqpe"),
    ("iqpe-m3-readout-hi",  "iqpe"),
]:
    for fmt in ["heatmap", "3d"]:
        qbp.plot_diff(
            log(name), method=method, plot_format=fmt,
            output_path=plt(f"{name}-diff-{fmt}"),
        )
        
def _grid_errors(log_paths, method):
    if isinstance(log_paths, str):
        log_paths = [log_paths]
    errors = []
    for log_path in log_paths:
        raw_data_path = log_path.replace(".json", ".raw-data.json")
        with open(raw_data_path) as f:
            data = json.load(f)
        cells = data["cells"]
        analytic = cells.get("analytic", {})
        sim      = cells.get(method, {})
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

print("\n" + "="*65)
print("NUMERICAL SUMMARY — error vs analytic across n_occ × t2")
print("="*65)

for label, raw_name, mit_name, method in [
    ("VQE + ZNE (gate noise)",   "vqe-raw-gate",        "vqe-zne-gate",       "vqe"),
    ("VQE + DD  (decoherence)",  "vqe-raw-decoherence", "vqe-dd-decoherence", "vqe"),
    ("IQPE + M3 (readout error)",["iqpe-raw-readout-lo", "iqpe-raw-readout-hi"],
                                 ["iqpe-m3-readout-lo",  "iqpe-m3-readout-hi"], "iqpe"),
]:
    print(f"\n{label}")
    print("-"*65)
    try:
        raw_names = raw_name if isinstance(raw_name, list) else [raw_name]
        mit_names = mit_name if isinstance(mit_name, list) else [mit_name]
        raw_errs = _grid_errors([log(n) for n in raw_names], method)
        mit_errs = _grid_errors([log(n) for n in mit_names], method)
        print(f"  {'Metric':<23} {'Raw':>10} {'Mitigated':>10} {'Improvement':>12}")
        for metric_label, r, m in [
            ("Mean abs error (MAE)", raw_errs.mean(), mit_errs.mean()),
            ("Median abs error",     np.median(raw_errs), np.median(mit_errs)),
            ("RMSE",                 _rmse(raw_errs), _rmse(mit_errs)),
            ("Max abs error",        raw_errs.max(), mit_errs.max()),
        ]:
            pct = 100 * (r - m) / r if r > 0 else float("nan")
            print(f"  {metric_label:<23} {r:>10.4f} {m:>10.4f} {pct:>+11.1f}%")
        helps = int((mit_errs < raw_errs).sum())
        hurts = int((mit_errs > raw_errs).sum())
        print(f"  Grid points:            {len(raw_errs)}")
        print(f"  Points mitigation helps: {helps} / {len(raw_errs)}")
        print(f"  Points mitigation hurts: {hurts} / {len(raw_errs)}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*65)
print("\nPlots saved to examples/plots/haldane/2x2/")