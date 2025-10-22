#!/usr/bin/env python3

import os, csv, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp

from quant_backend import (
    kitaev_pauli_hamiltonian,
    make_estimator,
    transpile_ansatz,
    vqe_min_eigenvalue,
)
from qiskit.circuit.library import EfficientSU2

## ensure output directory exists ##
def _ensuredir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

## exact ground energy from Pauli operator by dense diagonalization ##
def exact_ground_energy(H: SparsePauliOp) -> float:
    M = H.to_matrix(sparse=False)
    evals = np.linalg.eigvalsh(M)
    return float(np.min(evals).real)

def main():
    ap = argparse.ArgumentParser(description="Compare backend VQE vs exact for Kitaev chain")
    ap.add_argument("--outdir", type=str, default="out_backend_plots")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--Delta", type=float, default=0.25)
    ap.add_argument("--mu_min", type=float, default=-3.0)
    ap.add_argument("--mu_max", type=float, default=3.0)
    ap.add_argument("--num_mu", type=int, default=41)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_simple_noise", action="store_true")
    ap.add_argument("--maxiter", type=int, default=300)
    args = ap.parse_args()

    _ensuredir(args.outdir)

    ## build estimator and backend once ##
    noise = None
    if args.use_simple_noise:
        from qiskit_aer.noise import depolarizing_error, NoiseModel
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.001,1), ['rz','sx','x'])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.01,2), ['cx'])
        noise = nm

    shots = None if args.shots == 0 else args.shots
    estimator, backend = make_estimator(shots=shots, seed=args.seed, noise_model=noise)

    ## prepare ansatz once, transpiled to backend ISA ##
    ansatz = EfficientSU2(num_qubits=args.N, entanglement="linear", reps=args.reps)
    ansatz_isa = transpile_ansatz(ansatz, backend, optimization_level=3)

    mu_grid = np.linspace(args.mu_min, args.mu_max, args.num_mu)
    rows = []

    for mu in mu_grid:
        H = kitaev_pauli_hamiltonian(N=args.N, t=args.t, Delta=args.Delta, mu=float(mu))
        H_isa = H.apply_layout(layout=ansatz_isa.layout)

        e_vqe, _meta = vqe_min_eigenvalue(H_isa, ansatz_isa, estimator, maxiter=args.maxiter)
        e_exact = exact_ground_energy(H)  ## exact from un-transpiled H (physically identical) ##
        rows.append({"mu": float(mu), "E_exact": e_exact, "E_vqe": e_vqe, "abs_err": abs(e_vqe - e_exact)})

    ## write csv and json ##
    csv_path = os.path.join(args.outdir, "kitaev_vqe_vs_exact.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mu","E_exact","E_vqe","abs_err"])
        w.writeheader(); w.writerows(rows)

    json_path = os.path.join(args.outdir, "kitaev_vqe_vs_exact.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    ## plots ##
    mus = np.array([r["mu"] for r in rows])
    e_ex = np.array([r["E_exact"] for r in rows])
    e_vq = np.array([r["E_vqe"] for r in rows])
    err  = np.array([r["abs_err"] for r in rows])

    plt.figure()
    plt.plot(mus, e_ex, label="Exact", lw=2)
    plt.plot(mus, e_vq, label="VQE (backend)", lw=2, ls="--")
    plt.axvline(x=-2*args.t, color="k", lw=1, ls=":")
    plt.axvline(x=+2*args.t, color="k", lw=1, ls=":")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$E_0$")
    plt.title(f"Kitaev Chain Ground Energy: exact vs VQE (N={args.N}, Δ={args.Delta}, reps={args.reps}, shots={args.shots})")
    plt.legend()
    plt.tight_layout()
    eplot = os.path.join(args.outdir, "energy_vs_mu.png")
    plt.savefig(eplot, dpi=180); plt.close()

    plt.figure()
    plt.plot(mus, err, lw=2)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"|E_VQE - E_exact|")
    plt.title("Absolute Error vs $\mu$")
    plt.tight_layout()
    errplot = os.path.join(args.outdir, "abs_error_vs_mu.png")
    plt.savefig(errplot, dpi=180); plt.close()

    print(json.dumps({"csv": csv_path, "json": json_path, "energy_plot": eplot, "error_plot": errplot}, indent=2))

if __name__ == "__main__":
    main()
