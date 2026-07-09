"""Variational Quantum Eigensolver (VQE) simulation method and core solvers."""

from __future__ import annotations

import numpy as np

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, Estimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from quaph._core import _fmt_params, _make_simulator, logger
from quaph._mitigation import chain_measure, transform_circuit_chain
from quaph._method import Method, ParamSpec, SimulationMethod, register_method


# --------------------------------------------------------------------- solvers

def _vqe_initial_state(hamiltonian, ansatz, get_optimizer_fn, max_iters, backend=None) -> QuantumCircuit:
    simulator = _make_simulator(backend)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    with Session(backend=simulator) as session:
        estimator = Estimator(mode=session)
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history = {"iters": 0, "prev": None}

        def cost_func(params):
            if cost_history["iters"] >= max_iters and cost_history["prev"] is not None:
                return cost_history["prev"]
            pub = (ansatz_circuit, [hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = float(result[0].data.evs[0])
            cost_history["iters"] += 1
            cost_history["prev"] = energy
            return energy

        optimizer = get_optimizer_fn(max_iters)
        res = optimizer.minimize(cost_func, x0=x0)

    param_dict = dict(zip(ansatz.parameters, res.x))
    return ansatz.assign_parameters(param_dict)


def _expectation_from_counts(hamiltonian: SparsePauliOp, counts: dict) -> float:
    """Compute <H> from a (possibly M3-corrected) bitstring probability dict.

    Parameters
    ----------
    hamiltonian
        The qubit Hamiltonian as a SparsePauliOp.
    counts
        Dict mapping bitstring (e.g. ``"0101"``) to probability.
    """
    from qiskit.quantum_info import Statevector
    import re
    n = hamiltonian.num_qubits
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    energy = 0.0
    for pauli_term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        pauli_str = pauli_term.to_label()  # e.g. "IXYZ"
        term_exp = 0.0
        for bitstr, prob in counts.items():
            # Compute eigenvalue of this Pauli term for this bitstring
            # Only Z-type terms contribute; X/Y need off-diagonal elements
            # For a diagonal approximation (Z terms only):
            eigenval = 1.0
            for i, p in enumerate(reversed(pauli_str)):
                if p == "Z":
                    bit = int(bitstr[-(i + 1)]) if i < len(bitstr) else 0
                    eigenval *= (-1) ** bit
                elif p in ("X", "Y"):
                    eigenval = 0.0
                    break
            term_exp += prob * eigenval
        energy += float(coeff.real) * term_exp
    return energy


def _vqe_sparse(hamiltonian, ansatz, get_optimizer_fn, max_iters, rep, backend=None,
               label="", observable_qubit_ops=None, strategies=None,
               return_state: bool = False, seed: int | None = None):
    """Run VQE for a single repetition.

    Parameters
    ----------
    strategies
        List of active quaph._mitigation.MitigationStrategy
        instances (from self.mitigation_strategies), already
        calibrated. Each strategy's transform_circuit hook is applied
        once to the ansatz circuit; each strategy's measure hook wraps
        every expectation value evaluation (see quaph._mitigation).
    return_state
        If True, also return the optimized ansatz bound to its final
        parameters (to warm-start IQPE with a state that has real
        overlap with the ground state, instead of a bare HF reference).
        Mutually exclusive with observable_qubit_ops.
    seed
        If given, seeds every independent source of randomness in the VQE
        loop so that two calls with the same seed are fully reproducible
    """
    if seed is not None:
        np.random.seed(seed)
        from qiskit_algorithms.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    strategies = strategies or []
    simulator = _make_simulator(backend)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)
    ansatz_circuit = transform_circuit_chain(strategies, ansatz_circuit, simulator)

    # qiskit_aer.primitives.EstimatorV2 (not qiskit_ibm_runtime's) computes
    # the EXACT expectation value from the density matrix by default
    # (default_precision=0.0) rather than Monte Carlo shot sampling
    # correctly reflects the noise model (it's not stripped, just not
    # re-sampled with extra statistical noise on top), ~45x faster in
    # practice, and needs no shot-seeding since there's no sampling at all.
    estimator = AerEstimatorV2.from_backend(simulator)

    def _base_measure(circuit, op, params):
        pub = (circuit, [op], [params])
        result = estimator.run(pubs=[pub]).result()
        evs = result[0].data.evs
        return float(evs.flat[0]) if hasattr(evs, "flat") else float(evs[0])

    measure = chain_measure(strategies, _base_measure)

    x0 = 2 * np.pi * np.random.random(ansatz_circuit.num_parameters)
    cost_history = {"iters": 0, "cost_history": []}

    def cost_func(params):
        if cost_history["iters"] >= max_iters:
            return cost_history["cost_history"][-1]
        energy = measure(ansatz_circuit, hamiltonian, params)
        cost_history["iters"] += 1
        cost_history["cost_history"].append(energy)
        return energy

    optimizer = get_optimizer_fn(max_iters)
    res = optimizer.minimize(cost_func, x0=x0)
    energy = float(res.fun)
    logger.debug(f"VQE {label} = {energy}")

    if return_state:
        param_dict = dict(zip(ansatz_circuit.parameters, res.x))
        return energy, ansatz_circuit.assign_parameters(param_dict)

    if observable_qubit_ops is None:
        return energy

    optimal_params = np.asarray(res.x)
    observable_values = [measure(ansatz_circuit, op, optimal_params) for op in observable_qubit_ops]
    return energy, observable_values


def vqe_fermionic(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, get_optimizer_fn, get_vqe_ansatz_fn, mapper, max_iters, n_layers, rep, backend=None, observable_qubit_ops=None, strategies=None, return_state: bool = False, seed: int | None = None):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    ansatz = get_vqe_ansatz_fn(n_sites * spin, n_layers, n_occ, spin)
    label = f"({_fmt_params(lattice, n_occ, model_params, repetition=rep)})"
    return _vqe_sparse(
        qubit_hamiltonian, ansatz, get_optimizer_fn, max_iters, rep,
        backend=backend, label=label, observable_qubit_ops=observable_qubit_ops,
        strategies=strategies, return_state=return_state,
        seed=seed,
    )


def vqe_observable(model, lattice, n_sites, spin, n_occ, model_params, mapper, max_iters, n_layers, rep, observable, backend=None):
    obs = model.get_observable(observable)

    def sub_eval(sub_n_occ, observable_qubit_ops=None):
        return vqe_fermionic(
            lattice, n_sites, spin, sub_n_occ, model_params,
            model.fermionic_hamiltonian, model.get_optimizer,
            model.get_vqe_ansatz, mapper, max_iters, n_layers, rep,
            backend=backend, observable_qubit_ops=observable_qubit_ops,
        )

    if obs.quantum_composite is not None:
        n_orbitals = n_sites * spin
        return float(obs.quantum_composite(
            model, lattice, n_occ, model_params, mapper, n_orbitals, sub_eval,
        ))
    if observable == "E" or obs.quantum_operator is None:
        return float(sub_eval(n_occ))
    op_fermionic = obs.quantum_operator(model, lattice, **model_params)
    if op_fermionic is None:
        return 0.0
    op_qubit = mapper.map(op_fermionic)
    _, vals = sub_eval(n_occ, observable_qubit_ops=[op_qubit])
    return float(vals[0])


def vqe_bloch(k_tuple, model_params, bloch_hamiltonian_fn, get_optimizer_fn, max_iters, n_layers, rep, backend=None, strategies=None, seed: int | None = None):
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)
    ansatz = efficient_su2(hamiltonian.num_qubits, reps=n_layers)
    label = f"bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep})"
    return _vqe_sparse(hamiltonian, ansatz, get_optimizer_fn, max_iters, rep, backend=backend, label=label, strategies=strategies, seed=seed)


def vqe_operator(hamiltonian, get_vqe_ansatz_fn, get_optimizer_fn, max_iters, n_layers, rep, extremum="min", backend=None, label="", observable="E"):
    op = hamiltonian * -1 if extremum == "max" else hamiltonian
    ansatz = get_vqe_ansatz_fn(hamiltonian.num_qubits, n_layers, 0, 1)
    vqe_label = f"[{observable}] ({label}, rep={rep})" if label else f"[{observable}] (rep={rep})"
    energy = _vqe_sparse(op, ansatz, get_optimizer_fn, max_iters, rep, backend=backend, label=vqe_label)
    return -energy if extremum == "max" else energy


def vqe_other_benchmarks(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, get_vqe_ansatz_fn, mapper, max_iters, n_layers, vqe_reps=1, backend=None):
    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = get_vqe_ansatz_fn(n_sites * spin, n_layers, n_occ, spin)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    num_queries = qubit_hamiltonian.size * max_iters * vqe_reps
    full_circuit_depth = ansatz_circuit.depth()
    two_gate_circuit_depth = ansatz_circuit.depth(lambda x: x.operation.num_qubits == 2)

    logger.info(f"VQE other benchmarks ({_fmt_params(lattice, n_occ, model_params)}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth},{two_gate_circuit_depth}]")
    return num_queries, (full_circuit_depth, two_gate_circuit_depth)


def vqe_bloch_other_benchmarks(k_tuple, model_params, bloch_hamiltonian_fn, max_iters, n_layers, vqe_reps=1, backend=None):
    if backend:
        noise_model = NoiseModel.from_backend(backend)
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)
    ansatz = efficient_su2(hamiltonian.num_qubits, reps=n_layers)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    num_queries = hamiltonian.size * max_iters * vqe_reps
    full_circuit_depth = ansatz_circuit.depth()
    two_gate_circuit_depth = ansatz_circuit.depth(lambda x: x.operation.num_qubits == 2)

    logger.info(f"VQE bloch benchmarks (k={tuple(round(float(x), 3) for x in k_tuple)}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth},{two_gate_circuit_depth}]")
    return num_queries, (full_circuit_depth, two_gate_circuit_depth)


# ---------------------------------------------------------------------- method
@register_method
class VQEMethod(SimulationMethod):
    METHOD = Method.VQE
    LABEL = "VQE"
    PARAM_SPECS = [
        ParamSpec("iters", int, 100, "VQE optimizer iterations per repetition", metavar="N"),
        ParamSpec("layers", int, 1, "Number of ansatz layers (reps)", metavar="N"),
        ParamSpec("reps", int, 1, "Number of independent VQE repetitions", metavar="N"),
    ]
    # Dict-valued params used only on the --qubit-operator path (no model ansatz).
    EXTRA_PARAMS = ("ansatz", "optimizer")
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = True
    SUPPORTS_OPERATOR = True

    # ----------------------------------------------------------------- real space
    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        reps = []
        for rep in range(1, self.reps + 1):
            if observable == "E":
                energy = vqe_fermionic(
                    lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    model.fermionic_hamiltonian, model.get_optimizer,
                    model.get_vqe_ansatz, ctx.mapper, self.iters, self.layers, rep,
                    backend=backend, strategies=self.mitigation_strategies,
                    seed=ctx.cell_index * 1000 + rep,
                )
            else:
                energy = vqe_observable(
                    model, lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ctx.mapper, self.iters, self.layers, rep, observable,
                    backend=backend,
                )
            reps.append(float(energy))
        num_queries, (total, two_q) = vqe_other_benchmarks(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            model.fermionic_hamiltonian, model.get_vqe_ansatz, ctx.mapper,
            self.iters, self.layers, self.reps, backend=backend,
        )
        return {
            "repetitions": reps,
            "num_queries": num_queries,
            "circuit_depth": {"total": total, "two_qubit": two_q},
        }

    # -------------------------------------------------------------- band structure
    def compute_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                           backend, ctx):
        reps = []
        for rep in range(1, self.reps + 1):
            energy = vqe_bloch(
                k_tuple, cell_params, model.bloch_hamiltonian, model.get_optimizer,
                self.iters, self.layers, rep, backend=backend,
                strategies=self.mitigation_strategies, seed=ctx.cell_index * 1000 + rep,
            )
            reps.append(float(energy))
        num_queries, (total, two_q) = vqe_bloch_other_benchmarks(
            k_tuple, cell_params, model.bloch_hamiltonian,
            self.iters, self.layers, self.reps, backend=backend,
        )
        return {
            "repetitions": reps,
            "num_queries": num_queries,
            "circuit_depth": {"total": total, "two_qubit": two_q},
        }

    # ------------------------------------------------------------------- operator
    def compute_operator_cell(self, op, *, extremum, backend, label, observable: str = "E"):
        from quaph._yaml_model import (
            AnsatzSpec, OptimizerSpec,
            build_ansatz_factory, build_optimizer_factory,
        )
        ansatz_spec = (
            AnsatzSpec.model_validate(self.ansatz) if self.ansatz
            else AnsatzSpec(type="efficient_su2", kwargs={"reps": "@n_layers"},
                            initial_state_prefix="none")
        )
        get_vqe_ansatz = build_ansatz_factory(ansatz_spec, name="operator")
        optimizer_spec = (
            OptimizerSpec.model_validate(self.optimizer) if self.optimizer
            else OptimizerSpec(type="SPSA", kwargs={"maxiter": "@max_iters"})
        )
        get_optimizer = build_optimizer_factory(optimizer_spec, name="operator")
        reps = []
        for rep in range(1, self.reps + 1):
            energy = vqe_operator(
                op, get_vqe_ansatz, get_optimizer, self.iters, self.layers, rep,
                extremum, backend, label, observable,
            )
            reps.append(float(energy))
        return {"repetitions": reps}