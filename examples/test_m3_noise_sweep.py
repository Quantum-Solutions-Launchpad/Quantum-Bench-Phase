"""Test WHERE and WHY M3 helps: sweep readout error rate at fixed  moderate state-prep quality
"""
import math
import json
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

from quaph._registry import get_model
from quaph._vqe import vqe_fermionic
from quaph._iqpe import iqpe_fermionic
from quaph._mitigation_m3 import M3Strategy


def _readout_only(p_error):
    nm = NoiseModel()
    ro_err = ReadoutError([[1 - p_error, p_error], [p_error, 1 - p_error]])
    nm.add_all_qubit_readout_error(ro_err)
    return nm


MODEL = get_model("haldane")
LATTICE = (2, 2)
N_SITES, SPIN, N_OCC = 4, 2, 2
MODEL_PARAMS = {"t1": 1.0, "phi": math.pi / 4, "M": 0.1, "t2": 0.2}
ANALYTIC_E = -4.572339191022054

TIME, N_TROT, N_ITERS = 0.5, 4, 6
REPS = 10
WARM_START = dict(max_iters=120, n_layers=3)
NOISE_LEVELS = [0.02, 0.08, 0.15, 0.25, 0.35, 0.45]

mapper = MODEL.get_mapper(N_SITES, SPIN, N_OCC)

print("Preparing warm-start states (noiseless, shared across all noise levels)...", flush=True)
bound_circuits = []
for rep in range(REPS):
    np.random.seed(hash(("warmstart", rep)) % (2**31 - 1))
    vqe_energy, bound = vqe_fermionic(
        LATTICE, N_SITES, SPIN, N_OCC, MODEL_PARAMS, MODEL.fermionic_hamiltonian,
        MODEL.get_optimizer, MODEL.get_vqe_ansatz, mapper,
        WARM_START["max_iters"], WARM_START["n_layers"], rep=0,
        backend=None, return_state=True,
    )
    print(f"  rep={rep} vqe_energy={vqe_energy:.4f} (analytic {ANALYTIC_E:.4f})", flush=True)
    bound_circuits.append(bound)

results = {}
for p_error in NOISE_LEVELS:
    print(f"\n=== readout error rate: {p_error:.0%} ===", flush=True)
    backend = AerSimulator(noise_model=_readout_only(p_error))
    m3_strategy = M3Strategy()
    m3_strategy.calibrate(backend)

    raw_errs, m3_errs = [], []
    for rep, bound in enumerate(bound_circuits):
        e_raw, _ = iqpe_fermionic(
            LATTICE, N_SITES, SPIN, N_OCC, MODEL_PARAMS, MODEL.fermionic_hamiltonian, mapper,
            TIME, N_TROT, N_ITERS, rep,
            backend=backend, initial_state_override=bound,
        )
        e_m3, _ = iqpe_fermionic(
            LATTICE, N_SITES, SPIN, N_OCC, MODEL_PARAMS, MODEL.fermionic_hamiltonian, mapper,
            TIME, N_TROT, N_ITERS, rep,
            backend=backend, initial_state_override=bound, strategies=[m3_strategy],
        )
        raw_err = abs(e_raw - ANALYTIC_E)
        m3_err = abs(e_m3 - ANALYTIC_E)
        raw_errs.append(raw_err)
        m3_errs.append(m3_err)
        print(f"  rep={rep} raw={e_raw:.4f} (err {raw_err:.4f})  m3={e_m3:.4f} (err {m3_err:.4f})", flush=True)

    raw_errs = np.array(raw_errs)
    m3_errs = np.array(m3_errs)
    pct_mae = 100 * (raw_errs.mean() - m3_errs.mean()) / raw_errs.mean() if raw_errs.mean() > 0 else float("nan")
    pct_med = 100 * (np.median(raw_errs) - np.median(m3_errs)) / np.median(raw_errs) if np.median(raw_errs) > 0 else float("nan")
    results[p_error] = dict(
        raw_mae=float(raw_errs.mean()), m3_mae=float(m3_errs.mean()),
        raw_median=float(np.median(raw_errs)), m3_median=float(np.median(m3_errs)),
        mae_improvement_pct=float(pct_mae), median_improvement_pct=float(pct_med),
        helps=int((m3_errs < raw_errs).sum()), hurts=int((m3_errs > raw_errs).sum()),
        n=len(raw_errs),
    )
    r = results[p_error]
    print(f"  -> raw_mae={r['raw_mae']:.4f} m3_mae={r['m3_mae']:.4f} "
          f"MAE improvement={pct_mae:+.1f}%  helps={r['helps']}/{r['n']}", flush=True)

print("\n" + "=" * 78)
print("SUMMARY: M3 improvement vs readout error rate (fixed moderate state prep)")
print("=" * 78)
print(f"{'Noise':>8} {'Raw MAE':>10} {'M3 MAE':>10} {'MAE imp.':>10} {'Median imp.':>12} {'Helps':>8}")
for p_error in NOISE_LEVELS:
    r = results[p_error]
    print(f"{p_error:>7.0%} {r['raw_mae']:>10.4f} {r['m3_mae']:>10.4f} "
          f"{r['mae_improvement_pct']:>+9.1f}% {r['median_improvement_pct']:>+11.1f}%  "
          f"{r['helps']}/{r['n']}")
print("=" * 78)

out_path = "examples/logs/haldane/m3_noise_sweep_test.json"
with open(out_path, "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print(f"\nSaved to {out_path}")
