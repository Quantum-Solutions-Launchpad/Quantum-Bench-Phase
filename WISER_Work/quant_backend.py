#!/usr/bin/env python3

import argparse, json
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendEstimator  ## <-- key change

## minimal noise helper ##
def make_simple_noise_model(p1=0.001, p2=0.01):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rz','sx','x'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    return nm

## optional fake-device noise if you later install qiskit-ibm-runtime ##
def noise_from_fake(backend_name="FakeSherbrooke"):
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        fake = FakeSherbrooke()
        return NoiseModel.from_backend(fake)
    except Exception:
        return None

## build a backend + an Estimator bound to it (qiskit 1.4-friendly) ##
def make_estimator(shots=None, seed=1, noise_model=None):
    method = "statevector" if shots is None else "automatic"
    backend = AerSimulator(method=method)
    backend.set_options(seed_simulator=seed)

    if noise_model is not None:
        backend = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
        backend.set_options(seed_simulator=seed)

    ## BackendEstimator binds directly to the backend; shots go in options
    options = {"shots": shots} if shots is not None else {}
    est = BackendEstimator(backend=backend, options=options)
    return est, backend

## transpile ansatz to backend ISA for stability ##
def transpile_ansatz(ansatz, backend, optimization_level=3):
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=optimization_level)
    return pm.run(ansatz)

## generic VQE runner for any SparsePauliOp ##
def vqe_min_eigenvalue(H: SparsePauliOp, ansatz, estimator, maxiter=300):
    opt = SLSQP(maxiter=maxiter)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=opt)
    res = vqe.compute_minimum_eigenvalue(H)
    e0 = float(np.real(res.eigenvalue))
    meta = {
        "optimizer_evals": getattr(res, "optimizer_evals", None),
        "optimizer_result": str(getattr(res, "optimizer_result", ""))[:200]
    }
    return e0, meta

## example model adapter 1: Kitaev chain mapped to Pauli ##
def kitaev_pauli_hamiltonian(N=6, t=1.0, Delta=0.25, mu=0.0):
    coeffs, paulis = [], []
    for j in range(N-1):
        s = ["I"]*N; s[N-1-j]="X"; s[N-2-j]="X"; paulis.append("".join(s)); coeffs.append(-0.5*t)
        s = ["I"]*N; s[N-1-j]="Y"; s[N-2-j]="Y"; paulis.append("".join(s)); coeffs.append(-0.5*t)
        s = ["I"]*N; s[N-1-j]="X"; s[N-2-j]="Y"; paulis.append("".join(s)); coeffs.append(+0.5*Delta)
        s = ["I"]*N; s[N-1-j]="Y"; s[N-2-j]="X"; paulis.append("".join(s)); coeffs.append(-0.5*Delta)
    for j in range(N):
        s = ["I"]*N; s[N-1-j]="Z"; paulis.append("".join(s)); coeffs.append(0.5*mu)
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

## example model adapter 2: single-qubit Pauli vector for smoke tests ##
def pauli1_hamiltonian(hx=1.0, hy=0.0, hz=0.0):
    return SparsePauliOp.from_list([("X", hx), ("Y", hy), ("Z", hz)])

def main():
    ap = argparse.ArgumentParser(description="Model-agnostic VQE backend harness (Aer)")
    ap.add_argument("--model", type=str, default="kitaev", choices=["kitaev","pauli1"])
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--Delta", type=float, default=0.25)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--hx", type=float, default=1.0)
    ap.add_argument("--hy", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_fake_noise", action="store_true")
    ap.add_argument("--use_simple_noise", action="store_true")
    ap.add_argument("--maxiter", type=int, default=300)
    args = ap.parse_args()

    shots = None if args.shots == 0 else args.shots

    if args.model == "kitaev":
        H = kitaev_pauli_hamiltonian(N=args.N, t=args.t, Delta=args.Delta, mu=args.mu)
        num_qubits = args.N
    else:
        H = pauli1_hamiltonian(hx=args.hx, hy=args.hy, hz=args.hz)
        num_qubits = 1

    noise = None
    if args.use_fake_noise:
        noise = noise_from_fake()
    if noise is None and args.use_simple_noise:
        noise = make_simple_noise_model()

    estimator, backend = make_estimator(shots=shots, seed=args.seed, noise_model=noise)
    ans = EfficientSU2(num_qubits=num_qubits, entanglement="linear", reps=args.reps)
    ans_isa = transpile_ansatz(ans, backend, optimization_level=3)
    H_isa = H.apply_layout(layout=ans_isa.layout)

    e0, meta = vqe_min_eigenvalue(H_isa, ans_isa, estimator, maxiter=args.maxiter)

    out = {
        "model": args.model,
        "num_qubits": num_qubits,
        "reps": args.reps,
        "shots": shots,
        "seed": args.seed,
        "noise": ("fake" if args.use_fake_noise else ("simple" if args.use_simple_noise else "none")),
        "params": {"N":args.N,"t":args.t,"Delta":args.Delta,"mu":args.mu,"hx":args.hx,"hy":args.hy,"hz":args.hz},
        "E0_est": e0,
        "meta": meta
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
