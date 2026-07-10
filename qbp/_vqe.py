"""Variational Quantum Eigensolver (VQE) simulation method and core solvers."""

from __future__ import annotations

import numpy as np

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

from qbp._core import _fmt_params, _make_simulator, run_reps_parallel, logger
from qbp._backend import make_vqe_estimator
from qbp._method import Method, ParamSpec, SimulationMethod, register_method


def _isa(op, circ):
    return op.apply_layout(circ.layout) if circ.layout is not None else op


# --------------------------------------------------------------------- solvers
def _vqe_initial_state(hamiltonian, ansatz, get_optimizer_fn, max_iters, backend=None) -> QuantumCircuit:
    with make_vqe_estimator(backend) as est:
        ansatz_circuit = est.transpile(ansatz)
        isa_hamiltonian = _isa(hamiltonian, ansatz_circuit)
        estimator = est.estimator
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
        cost_history = {"iters": 0, "prev": None}

        def cost_func(params):
            if cost_history["iters"] >= max_iters and cost_history["prev"] is not None:
                return cost_history["prev"]
            pub = (ansatz_circuit, [isa_hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = float(result[0].data.evs[0])
            cost_history["iters"] += 1
            cost_history["prev"] = energy
            return energy

        optimizer = get_optimizer_fn(max_iters)
        res = optimizer.minimize(cost_func, x0=x0)

    param_dict = dict(zip(ansatz.parameters, res.x))
    return ansatz.assign_parameters(param_dict)


def _vqe_sparse(hamiltonian, ansatz, get_optimizer_fn, max_iters, rep, backend=None, label="", observable_qubit_ops=None):
    with make_vqe_estimator(backend) as est:
        ansatz_circuit = est.transpile(ansatz)
        isa_hamiltonian = _isa(hamiltonian, ansatz_circuit)
        estimator = est.estimator
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)

        cost_history = {"iters": 0, "cost_history": []}

        def cost_func(params):
            if cost_history["iters"] >= max_iters:
                return cost_history["cost_history"][-1]
            pub = (ansatz_circuit, [isa_hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]
            cost_history["iters"] += 1
            cost_history["cost_history"].append(energy)
            return energy

        optimizer = get_optimizer_fn(max_iters)
        res = optimizer.minimize(cost_func, x0=x0)
        energy = float(res.fun)
        logger.debug(f"VQE {label} = {energy}")

        if observable_qubit_ops is None:
            return energy

        optimal_params = np.asarray(res.x)
        observable_values = []
        for op in observable_qubit_ops:
            pub = (ansatz_circuit, [_isa(op, ansatz_circuit)], [optimal_params])
            result = estimator.run(pubs=[pub]).result()
            observable_values.append(float(result[0].data.evs[0]))
        return energy, observable_values


def vqe_fermionic(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, get_optimizer_fn, get_vqe_ansatz_fn, mapper, max_iters, n_layers, rep, backend=None, observable_qubit_ops=None):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    ansatz = get_vqe_ansatz_fn(n_sites * spin, n_layers, n_occ, spin)
    label = f"({_fmt_params(lattice, n_occ, model_params, repetition=rep)})"
    return _vqe_sparse(
        qubit_hamiltonian, ansatz, get_optimizer_fn, max_iters, rep,
        backend=backend, label=label, observable_qubit_ops=observable_qubit_ops,
    )


def vqe_observable(model, lattice, n_sites, spin, n_occ, model_params, mapper, max_iters, n_layers, rep, observable, backend=None, fermionic_hamiltonian_fn=None):
    obs = model.get_observable(observable)
    fermionic_hamiltonian_fn = fermionic_hamiltonian_fn or model.fermionic_hamiltonian

    def sub_eval(sub_n_occ, observable_qubit_ops=None):
        return vqe_fermionic(
            lattice, n_sites, spin, sub_n_occ, model_params,
            fermionic_hamiltonian_fn, model.get_optimizer,
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


def vqe_bloch(k_tuple, model_params, bloch_hamiltonian_fn, get_optimizer_fn, max_iters, n_layers, rep, backend=None):
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)
    ansatz = efficient_su2(hamiltonian.num_qubits, reps=n_layers)
    label = f"bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep})"
    return _vqe_sparse(hamiltonian, ansatz, get_optimizer_fn, max_iters, rep, backend=backend, label=label)


def vqe_operator(hamiltonian, get_vqe_ansatz_fn, get_optimizer_fn, max_iters, n_layers, rep, extremum="min", backend=None, label="", observable="E"):
    op = hamiltonian * -1 if extremum == "max" else hamiltonian
    ansatz = get_vqe_ansatz_fn(hamiltonian.num_qubits, n_layers, 0, 1)
    vqe_label = f"[{observable}] ({label}, rep={rep})" if label else f"[{observable}] (rep={rep})"
    energy = _vqe_sparse(op, ansatz, get_optimizer_fn, max_iters, rep, backend=backend, label=vqe_label)
    return -energy if extremum == "max" else energy


def vqe_other_benchmarks(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, get_vqe_ansatz_fn, mapper, max_iters, n_layers, vqe_reps=1, backend=None):
    simulator = _make_simulator(backend)

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
    simulator = _make_simulator(backend)

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
        ferm_fn = ctx.fermionic_hamiltonian_fn or model.fermionic_hamiltonian

        def run_one_rep(rep):
            if observable == "E":
                energy = vqe_fermionic(
                    lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ferm_fn, model.get_optimizer,
                    model.get_vqe_ansatz, ctx.mapper, self.iters, self.layers, rep,
                    backend=backend,
                )
            else:
                energy = vqe_observable(
                    model, lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ctx.mapper, self.iters, self.layers, rep, observable,
                    backend=backend, fermionic_hamiltonian_fn=ferm_fn,
                )
            return float(energy)

        reps = run_reps_parallel(self.reps, run_one_rep)
        num_queries, (total, two_q) = vqe_other_benchmarks(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            ferm_fn, model.get_vqe_ansatz, ctx.mapper,
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
        def run_one_rep(rep):
            energy = vqe_bloch(
                k_tuple, cell_params, model.bloch_hamiltonian, model.get_optimizer,
                self.iters, self.layers, rep, backend=backend,
            )
            return float(energy)

        reps = run_reps_parallel(self.reps, run_one_rep)
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
        from qbp._yaml_model import (
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

        def run_one_rep(rep):
            energy = vqe_operator(
                op, get_vqe_ansatz, get_optimizer, self.iters, self.layers, rep,
                extremum, backend, label, observable,
            )
            return float(energy)

        reps = run_reps_parallel(self.reps, run_one_rep)
        return {"repetitions": reps}
