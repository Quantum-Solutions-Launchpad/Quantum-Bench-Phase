"""Variational Quantum Eigensolver (VQE) simulation method and core solvers."""

from __future__ import annotations

import numpy as np

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

from qbp._backend import make_vqe_estimator
from qbp._core import _fmt_params, _make_simulator, logger
from qbp._mitigation import chain_measure, transform_circuit_chain
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


def _strip_x_prefix(circuit: QuantumCircuit) -> QuantumCircuit:
    body = circuit.copy_empty_like()
    in_prefix = True
    for instruction in circuit.data:
        if in_prefix and instruction.operation.name == "x":
            continue
        in_prefix = False
        body.append(instruction)
    return body


def _warm_start_ansatz(fermionic_hamiltonian, ansatz, n_orbitals, n_occ, mapper):
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from qbp._givens import free_fermion_prep, one_body_matrix

    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(
            "The non-interacting warm start requires the Jordan-Wigner mapper."
        )
    h = one_body_matrix(fermionic_hamiltonian, n_orbitals)
    prep = free_fermion_prep(h, n_occ)
    return prep.compose(_strip_x_prefix(ansatz))


def _vqe_sparse(hamiltonian, ansatz, get_optimizer_fn, max_iters, rep, backend=None,
               label="", observable_qubit_ops=None, strategies=None,
               return_state: bool = False, seed: int | None = None,
               warm_start: bool = False):
    """Run VQE for a single repetition.

    Parameters
    ----------
    strategies
        List of active :class:`~qbp._mitigation.MitigationStrategy`
        instances (from ``self.mitigation_strategies``), already
        calibrated. Each strategy's ``transform_circuit`` hook is applied
        once to the ansatz circuit; each strategy's ``measure`` hook wraps
        every expectation-value evaluation (see qbp._mitigation).
    return_state
        If True, also return the optimized ansatz bound to its final
        parameters (e.g. to warm-start IQPE with a state that has real
        overlap with the ground state, instead of a bare HF reference).
        Mutually exclusive with observable_qubit_ops.
    seed
        If given, seeds every independent source of randomness in the VQE
        loop so that two calls with the same seed are fully reproducible:
        (1) numpy's global RNG, for x0; (2) qiskit_algorithms'
        algorithm_globals RNG, since SPSA's bernoulli_perturbation draws
        from algorithm_globals.random — a separate generator
        np.random.seed() never touches, so without this two "identically
        seeded" runs still diverge because SPSA's own perturbation-direction
        randomness at every iteration is uncontrolled. quaph.run() dispatches
        grid cells to worker processes via joblib (loky/spawn on macOS),
        which do NOT inherit the parent process's seeded RNG state, so a
        top-level np.random.seed() call in a driver script has no effect on
        either of these here.
    """
    if seed is not None:
        np.random.seed(seed)
        from qiskit_algorithms.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    strategies = strategies or []

    # Separate from make_vqe_estimator's own internal simulator: DD's
    # transform_circuit needs a backend-like object to schedule against
    # (InstructionDurations.from_backend), and make_vqe_estimator doesn't
    # expose its internal simulator as part of its public interface.
    # _make_simulator is cheap/deterministic, so calling it again here is
    # harmless.
    simulator = _make_simulator(backend)

    with make_vqe_estimator(backend) as est:
        ansatz_circuit = est.transpile(ansatz)
        ansatz_circuit = transform_circuit_chain(strategies, ansatz_circuit, simulator)
        estimator = est.estimator

        def _base_measure(circuit, op, params):
            pub = (circuit, [_isa(op, circuit)], [params])
            result = estimator.run(pubs=[pub]).result()
            evs = result[0].data.evs
            return float(evs.flat[0]) if hasattr(evs, "flat") else float(evs[0])

        measure = chain_measure(strategies, _base_measure)

        if warm_start:
            x0 = np.zeros(ansatz_circuit.num_parameters)
        else:
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


def vqe_fermionic(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, get_optimizer_fn, get_vqe_ansatz_fn, mapper, max_iters, n_layers, rep, backend=None, observable_qubit_ops=None, strategies=None, return_state: bool = False, seed: int | None = None, warm_start: bool = False):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    ansatz = get_vqe_ansatz_fn(n_sites * spin, n_layers, n_occ, spin)
    if warm_start:
        ansatz = _warm_start_ansatz(
            fermionic_hamiltonian, ansatz, n_sites * spin, n_occ, mapper
        )
    label = f"({_fmt_params(lattice, n_occ, model_params, repetition=rep)})"
    return _vqe_sparse(
        qubit_hamiltonian, ansatz, get_optimizer_fn, max_iters, rep,
        backend=backend, label=label, observable_qubit_ops=observable_qubit_ops,
        strategies=strategies, return_state=return_state,
        seed=seed, warm_start=warm_start,
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


def _vqe_measured_ansatz_and_groups(qubit_hamiltonian, ansatz):
    """A measured copy of the ansatz plus the number of commuting Pauli groups.

    The Estimator measures one circuit per commuting group of the Hamiltonian,
    so the group count is the per-evaluation circuit multiplier for cost.
    """
    measured = ansatz.copy()
    measured.measure_all()
    return measured, len(qubit_hamiltonian.group_commuting())


def vqe_cell_seconds(lattice, n_sites, spin, n_occ, model_params,
                     fermionic_hamiltonian_fn, get_vqe_ansatz_fn, mapper,
                     max_iters, n_layers, reps, backend, shots,
                     warm_start: bool = False):
    from qbp._backend import circuit_qpu_seconds

    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    ansatz = get_vqe_ansatz_fn(n_sites * spin, n_layers, n_occ, spin)
    if warm_start:
        ansatz = _warm_start_ansatz(
            fermionic_hamiltonian, ansatz, n_sites * spin, n_occ, mapper
        )
    measured, n_groups = _vqe_measured_ansatz_and_groups(qubit_hamiltonian, ansatz)
    return circuit_qpu_seconds(backend, measured, shots) * n_groups * max_iters * reps


def vqe_bloch_seconds(k_tuple, model_params, bloch_hamiltonian_fn,
                      max_iters, n_layers, reps, backend, shots):
    from qbp._backend import circuit_qpu_seconds

    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)
    ansatz = efficient_su2(hamiltonian.num_qubits, reps=n_layers)
    measured, n_groups = _vqe_measured_ansatz_and_groups(hamiltonian, ansatz)
    return circuit_qpu_seconds(backend, measured, shots) * n_groups * max_iters * reps


def vqe_operator_seconds(hamiltonian, get_vqe_ansatz_fn, max_iters, n_layers, reps, backend, shots):
    from qbp._backend import circuit_qpu_seconds

    ansatz = get_vqe_ansatz_fn(hamiltonian.num_qubits, n_layers, 0, 1)
    measured, n_groups = _vqe_measured_ansatz_and_groups(hamiltonian, ansatz)
    return circuit_qpu_seconds(backend, measured, shots) * n_groups * max_iters * reps


# ---------------------------------------------------------------------- method
@register_method
class VQEMethod(SimulationMethod):
    METHOD = Method.VQE
    LABEL = "VQE"
    PARAM_SPECS = [
        ParamSpec("iters", int, 100, "VQE optimizer iterations per repetition", metavar="N"),
        ParamSpec("layers", int, 1, "Number of ansatz layers (reps)", metavar="N"),
        ParamSpec("reps", int, 1, "Number of independent VQE repetitions", metavar="N"),
        ParamSpec("warm_start", bool, False,
                  "Non-interacting warm start: prepare the ground state of the "
                  "one-body part of the Hamiltonian via a Givens-rotation "
                  "network (O(N^3) classical work) and start the variational "
                  "parameters at zero", is_flag=True),
    ]
    # Dict-valued params used only on the --qubit-operator path (no model ansatz).
    EXTRA_PARAMS = ("ansatz", "optimizer")
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = True
    SUPPORTS_OPERATOR = True

    def _ansatz_fn(self, model):
        if not self.ansatz:
            return model.get_vqe_ansatz
        from qbp._yaml_model import AnsatzSpec, build_ansatz_factory

        return build_ansatz_factory(
            AnsatzSpec.model_validate(self.ansatz), name="override"
        )

    def _optimizer_fn(self, model):
        if not self.optimizer:
            return model.get_optimizer
        from qbp._yaml_model import OptimizerSpec, build_optimizer_factory

        return build_optimizer_factory(
            OptimizerSpec.model_validate(self.optimizer), name="override"
        )

    # ----------------------------------------------------------------- real space
    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        ferm_fn = ctx.fermionic_hamiltonian_fn or model.fermionic_hamiltonian
        get_ansatz = self._ansatz_fn(model)
        get_optimizer = self._optimizer_fn(model)
        reps = []
        for rep in range(1, self.reps + 1):
            if observable == "E":
                energy = vqe_fermionic(
                    lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ferm_fn, get_optimizer,
                    get_ansatz, ctx.mapper, self.iters, self.layers, rep,
                    backend=backend, strategies=self.mitigation_strategies,
                    seed=ctx.cell_index * 1000 + rep,
                    warm_start=self.warm_start,
                )
            else:
                energy = vqe_observable(
                    model, lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ctx.mapper, self.iters, self.layers, rep, observable,
                    backend=backend, fermionic_hamiltonian_fn=ferm_fn,
                )
            reps.append(float(energy))
        num_queries, (total, two_q) = vqe_other_benchmarks(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            ferm_fn, get_ansatz, ctx.mapper,
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
        reps = []
        for rep in range(1, self.reps + 1):
            energy = vqe_operator(
                op, get_vqe_ansatz, get_optimizer, self.iters, self.layers, rep,
                extremum, backend, label, observable,
            )
            reps.append(float(energy))
        return {"repetitions": reps}

    def _estimate_shots(self, shots):
        from qbp._backend import _VQE_ESTIMATOR_SHOTS

        return _VQE_ESTIMATOR_SHOTS if shots is None else shots

    def estimate_cell(self, model, lattice, n_occ, cell_params, observable, *,
                      backend, ctx, shots):
        ferm_fn = ctx.fermionic_hamiltonian_fn or model.fermionic_hamiltonian
        return vqe_cell_seconds(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            ferm_fn, self._ansatz_fn(model), ctx.mapper,
            self.iters, self.layers, self.reps, backend, self._estimate_shots(shots),
            warm_start=self.warm_start,
        )

    def estimate_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                            backend, ctx, shots):
        return vqe_bloch_seconds(
            k_tuple, cell_params, model.bloch_hamiltonian,
            self.iters, self.layers, self.reps, backend, self._estimate_shots(shots),
        )

    def estimate_operator_cell(self, op, *, extremum, backend, observable="E", shots):
        from qbp._yaml_model import AnsatzSpec, build_ansatz_factory

        ansatz_spec = (
            AnsatzSpec.model_validate(self.ansatz) if self.ansatz
            else AnsatzSpec(type="efficient_su2", kwargs={"reps": "@n_layers"},
                            initial_state_prefix="none")
        )
        get_vqe_ansatz = build_ansatz_factory(ansatz_spec, name="operator")
        return vqe_operator_seconds(
            op, get_vqe_ansatz, self.iters, self.layers, self.reps, backend,
            self._estimate_shots(shots),
        )