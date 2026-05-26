import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, Estimator

from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit_aer.noise import NoiseModel

import sys
from loguru import logger


def setup_logging():
    fmt_console_info = "[<bold><green>{time:HH:mm:ss}</green></bold>] <white>{message}</white>"
    fmt_console_debug = "[<dim><white>{time:HH:mm:ss}</white></dim>] <dim>{message}</dim>"

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format=fmt_console_info,
        filter=lambda record: record["level"].name == "INFO",
    )
    logger.add(
        sys.stdout,
        level="DEBUG",
        colorize=True,
        format=fmt_console_debug,
        filter=lambda record: record["level"].name == "DEBUG",
    )

    return logger


def _fmt_params(lattice, n_occ, model_params=None, **extra):
    parts = [f"lattice={tuple(lattice)}", f"n_occ={n_occ}"]
    for k, v in (model_params or {}).items():
        parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def resolve_sweep(param: str, range_args, n_orbitals: int, momentum_axes: tuple[str, ...] = ()):
    if param == "n_occ":
        if range_args is None:
            vals = list(range(n_orbitals + 1))
        else:
            lo, hi, st = range_args
            vals = list(range(int(lo), int(hi) + 1, max(1, int(st))))
        return vals, r"$N_{\text{occ}}$", "n_occ"
    if param in momentum_axes:
        if range_args is None:
            lo, hi, st = -np.pi, np.pi, np.pi / 50
        else:
            lo, hi, st = range_args
        vals = list(np.arange(lo, hi + st / 2, st))
        return vals, param, "momentum"
    if range_args is None:
        raise ValueError(
            f"A sweep range is required when the sweep parameter is '{param}' (not 'n_occ')."
        )
    lo, hi, st = range_args
    vals = list(np.arange(lo, hi + st / 2, st))
    return vals, param, "parameter"


def _hf_initial_state(n_sites: int, spin: int, n_occ: int, mapper):
    if spin == 2:
        return HartreeFock(n_sites, (n_occ // 2 + n_occ % 2, n_occ // 2), mapper)
    num_modes = n_sites
    label = " ".join(f"+_{i}" for i in range(n_occ))
    bitstr_op = FermionicOp({label: 1.0} if label else {"": 1.0}, num_spin_orbitals=num_modes)
    qubit_op = mapper.map(bitstr_op)
    bits = qubit_op.paulis.x[0]
    qc = QuantumCircuit(len(bits))
    for i, bit in enumerate(bits):
        if bit:
            qc.x(i)
    return qc


def iqpe_estimate(unitary: QuantumCircuit, state_preparation: QuantumCircuit, num_iterations: int, sampler: Sampler, lattice, n_occ: int, rep: int, model_params: dict | None = None):
    omega_coef = 0
    iteration_phases = []

    for k in range(num_iterations, 0, -1):
        omega_coef /= 2

        qc = construct_iqpe_circuit(unitary, state_preparation, k, -2 * np.pi * omega_coef)

        sampler_job = sampler.run([qc])
        result = sampler_job.result().quasi_dists[0]
        x = 1 if result.get(1, 0) > result.get(0, 0) else 0

        omega_coef = omega_coef + x / 2
        iteration_phases.append(omega_coef)

        logger.debug(f"IQPE ({_fmt_params(lattice, n_occ, model_params, repetition=rep, iteration=num_iterations-k+1)}) = {omega_coef}")

    return omega_coef, iteration_phases


def construct_iqpe_circuit(unitary: QuantumCircuit, state_preparation: QuantumCircuit, k: int, omega: float):
    phase_register = QuantumRegister(1, name="a")
    eigenstate_register = QuantumRegister(unitary.num_qubits, name="q")

    qc = QuantumCircuit(eigenstate_register)
    qc.add_register(phase_register)
    qc.append(state_preparation, eigenstate_register)

    qc.h(phase_register[0])
    for _ in range(2 ** (k - 1)):
        qc = qc.compose(unitary.control(), [unitary.num_qubits] + list(range(0, unitary.num_qubits)))
    qc.p(omega, phase_register[0])
    qc.h(phase_register[0])

    c = ClassicalRegister(1, name="c")
    qc.add_register(c)
    qc.measure(phase_register, c)

    return qc


def analytic_bands(model, k_tuple, model_params, observable: str = "E"):
    H = model.bloch_hamiltonian(*k_tuple, **model_params)
    eigvals, eigvecs = np.linalg.eigh(H)
    obs = model.get_observable(observable)
    if obs.analytic_bloch is None:
        raise ValueError(
            f"Observable '{observable}' on model '{model.name}' has no analytic_bloch backend."
        )
    result = obs.analytic_bloch(model, k_tuple, H, eigvals, eigvecs, model_params)
    k_str = tuple(round(float(x), 3) for x in k_tuple)
    logger.info(f"Analytic [{observable}] (k={k_str}, {_fmt_params((), 0, model_params).split(', ', 2)[-1]}) = {result}")
    return result


def vqe_bloch(k_tuple, model_params, bloch_hamiltonian_fn, get_optimizer_fn, max_iters, n_layers, rep, backend=None):
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)

    if backend:
        noise_model = NoiseModel.from_backend(backend)
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = efficient_su2(hamiltonian.num_qubits, reps=n_layers)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    with Session(backend=simulator) as session:
        estimator = Estimator(mode=session)
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)

        cost_history = {"iters": 0, "cost_history": []}

        def cost_func(params):
            if cost_history["iters"] >= max_iters:
                return cost_history["cost_history"][-1]
            pub = (ansatz_circuit, [hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]
            cost_history["iters"] += 1
            cost_history["cost_history"].append(energy)
            return energy

        optimizer = get_optimizer_fn(max_iters)
        res = optimizer.minimize(cost_func, x0=x0)
        logger.debug(f"VQE bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep}) = {float(res.fun)}")
        return float(res.fun)


def iqpe_bloch(k_tuple, model_params, bloch_hamiltonian_fn, time_param, n_trot, n_iters, rep, backend=None):
    np.seterr(all='ignore')
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)

    n_qubits = hamiltonian.num_qubits
    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(hamiltonian, time=time_param, synthesis=st)
    initial = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        initial.h(q)

    if backend:
        noise_model = NoiseModel.from_backend(backend)
        sampler = Sampler(
            backend_options={
                "noise_model": noise_model,
                "basis_gates": noise_model.basis_gates,
            }
        )
    else:
        sampler = Sampler()

    omega_coef = 0
    iter_phases = []
    for k in range(n_iters, 0, -1):
        omega_coef /= 2
        qc = construct_iqpe_circuit(evolution, initial, k, -2 * np.pi * omega_coef)
        sampler_job = sampler.run([qc])
        result = sampler_job.result().quasi_dists[0]
        x = 1 if result.get(1, 0) > result.get(0, 0) else 0
        omega_coef = omega_coef + x / 2
        iter_phases.append(omega_coef)
        logger.debug(f"IQPE bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep}, iteration={n_iters-k+1}) phase={omega_coef}")

    res = float(-2 * np.pi * omega_coef / time_param)
    iter_energies = [float(-2 * np.pi * p / time_param) for p in iter_phases]
    logger.debug(f"IQPE bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep}) = {res}")
    return res, iter_energies


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


def iqpe_bloch_other_benchmarks(k_tuple, model_params, bloch_hamiltonian_fn, time_param, n_trot, n_iters, iqpe_reps, backend=None):
    np.seterr(all='ignore')
    if backend:
        noise_model = NoiseModel.from_backend(backend)
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)
    n_qubits = hamiltonian.num_qubits

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(hamiltonian, time=time_param, synthesis=st)
    initial = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        initial.h(q)

    full_circuit_depth = two_gate_circuit_depth = 0
    for k in range(n_iters, 0, -1):
        qc = construct_iqpe_circuit(evolution, initial, k, -2 * np.pi)
        qc = transpile(qc, backend=simulator, optimization_level=3)
        full_circuit_depth += qc.depth()
        two_gate_circuit_depth += qc.depth(lambda x: x.operation.num_qubits == 2)

    num_queries = hamiltonian.size * iqpe_reps * n_trot * n_iters

    logger.info(f"IQPE bloch benchmarks (k={tuple(round(float(x), 3) for x in k_tuple)}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth // n_iters},{two_gate_circuit_depth // n_iters}]")
    return num_queries, (full_circuit_depth // n_iters, two_gate_circuit_depth // n_iters)


def analytic(model, lattice, n_occ, model_params, observable: str = "E"):
    H = model._build_H_matrix(lattice, **model_params)
    eigvals, eigvecs = np.linalg.eigh(H)
    obs = model.get_observable(observable)
    result = float(obs.analytic(model, lattice, H, eigvals, eigvecs, n_occ, model_params))
    logger.info(f"Analytic [{observable}] ({_fmt_params(lattice, n_occ, model_params)}) = {result}")
    return result


def vqe(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, get_optimizer_fn, get_vqe_ansatz_fn, mapper, max_iters, n_layers, rep, backend=None, observable_qubit_ops=None):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = get_vqe_ansatz_fn(n_sites * spin, n_layers, n_occ, spin)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    with Session(backend=simulator) as session:
        estimator = Estimator(mode=session)
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)

        cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }

        def cost_func(params):
            if cost_history_dict["iters"] >= max_iters:
                return cost_history_dict["cost_history"][-1]

            pub = (ansatz_circuit, [qubit_hamiltonian], [params])
            result = estimator.run(pubs=[pub]).result()
            energy = result[0].data.evs[0]

            cost_history_dict["iters"] += 1
            cost_history_dict["prev_vector"] = params
            cost_history_dict["cost_history"].append(energy)

            return energy

        optimizer = get_optimizer_fn(max_iters)
        res = optimizer.minimize(cost_func, x0=x0)
        energy = float(res.fun)
        optimal_params = np.asarray(res.x)

        logger.debug(f"VQE ({_fmt_params(lattice, n_occ, model_params, repetition=rep)}) = {energy}")

        if observable_qubit_ops is None:
            return energy

        observable_values = []
        for op in observable_qubit_ops:
            pub = (ansatz_circuit, [op], [optimal_params])
            result = estimator.run(pubs=[pub]).result()
            observable_values.append(float(result[0].data.evs[0]))
        return energy, observable_values


def vqe_observable(model, lattice, n_sites, spin, n_occ, model_params, mapper, max_iters, n_layers, rep, observable, backend=None):
    obs = model.get_observable(observable)

    def sub_eval(sub_n_occ, observable_qubit_ops=None):
        return vqe(
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


def iqpe_observable(model, lattice, n_sites, spin, n_occ, model_params, mapper, time_param, n_trot, n_iters, rep, observable, backend=None):
    obs = model.get_observable(observable)

    def sub_eval(sub_n_occ, observable_qubit_ops=None):
        if observable_qubit_ops is not None:
            raise NotImplementedError(
                "IQPE only supports observables whose composite uses energies; "
                "operator-measurement observables require VQE."
            )
        energy, _ = iqpe(
            lattice, n_sites, spin, sub_n_occ, model_params,
            model.fermionic_hamiltonian, mapper, time_param, n_trot, n_iters, rep,
            backend=backend,
        )
        return energy

    if obs.quantum_composite is not None:
        n_orbitals = n_sites * spin
        return float(obs.quantum_composite(
            model, lattice, n_occ, model_params, mapper, n_orbitals, sub_eval,
        )), []
    if observable == "E":
        return iqpe(
            lattice, n_sites, spin, n_occ, model_params,
            model.fermionic_hamiltonian, mapper, time_param, n_trot, n_iters, rep,
            backend=backend,
        )
    raise NotImplementedError(
        f"IQPE cannot directly measure observable '{observable}' "
        f"(requires operator measurement on the prepared state). Use VQE instead."
    )


def iqpe(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, rep, backend=None):
    np.seterr(all='ignore')

    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=time_param, synthesis=st)
    initial = _hf_initial_state(n_sites, spin, n_occ, mapper)

    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        sampler = Sampler(
            backend_options={
                "noise_model": noise_model,
                "basis_gates": noise_model.basis_gates,
            }
        )
    else:
        sampler = Sampler()

    phase, iteration_phases = iqpe_estimate(evolution, initial, n_iters, sampler, lattice, n_occ, rep, model_params)
    res = float(-2 * np.pi * phase / time_param)
    iteration_energies = [float(-2 * np.pi * p / time_param) for p in iteration_phases]

    logger.debug(f"IQPE ({_fmt_params(lattice, n_occ, model_params, repetition=rep)}) = {res}")
    return res, iteration_energies


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


def iqpe_other_benchmarks(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, iqpe_reps, backend=None):
    np.seterr(all='ignore')
    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=time_param, synthesis=st)
    initial = _hf_initial_state(n_sites, spin, n_occ, mapper)

    full_circuit_depth = two_gate_circuit_depth = 0
    for k in range(n_iters, 0, -1):
        qc = construct_iqpe_circuit(evolution, initial, k, -2 * np.pi)
        qc = transpile(qc, backend=simulator, optimization_level=3)
        full_circuit_depth += qc.depth()
        two_gate_circuit_depth += qc.depth(lambda x: x.operation.num_qubits == 2)
        logger.debug(f"IQPE other benchmarks ({_fmt_params(lattice, n_occ, model_params, iteration=n_iters-k+1)}): circuit_depth=[{qc.depth(), qc.depth(lambda x: x.operation.num_qubits == 2)}]")

    num_queries = qubit_hamiltonian.size * iqpe_reps * n_trot * n_iters

    logger.info(f"IQPE other benchmarks ({_fmt_params(lattice, n_occ, model_params)}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth // n_iters},{two_gate_circuit_depth // n_iters}]")
    return num_queries, (full_circuit_depth // n_iters, two_gate_circuit_depth // n_iters)
