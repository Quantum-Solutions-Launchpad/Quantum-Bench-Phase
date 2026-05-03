import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import excitation_preserving, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit_ibm_runtime import Session, Estimator

from qiskit_nature.second_q.circuit.library import HartreeFock

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit_aer.noise import NoiseModel

import sys
from loguru import logger


def setup_logging(debug_enabled: bool = True):
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
    if debug_enabled:
        logger.add(
            sys.stdout,
            level="DEBUG",
            colorize=True,
            format=fmt_console_debug,
            filter=lambda record: record["level"].name == "DEBUG",
        )

    return logger


def resolve_sweep(param: str, range_args, n_sites: int, spin: int = 2):
    if param == "n_occ":
        if range_args is None:
            vals = list(range(spin * n_sites + 1))
        else:
            lo, hi, st = range_args
            vals = list(range(int(lo), int(hi) + 1, max(1, int(st))))
        return vals, r"$N_{\text{occ}}$", True
    else:
        if range_args is None:
            raise ValueError(
                f"A sweep range is required when the sweep parameter is '{param}' (not 'n_occ')."
            )
        lo, hi, st = range_args
        vals = list(np.arange(lo, hi + st / 2, st))
        return vals, param, False


def EP_ansatz(n_sites: int, n_layers: int, n_occ: int):
    spin = 2
    ansatz = QuantumCircuit(n_sites * spin)
    for i in range(n_occ):
        ansatz.x(i)
    ansatz.compose(excitation_preserving(n_sites * spin, "fsim", "linear", reps=n_layers), inplace=True)
    return ansatz


def iqpe_estimate(unitary: QuantumCircuit, state_preparation: QuantumCircuit, num_iterations: int, sampler: Sampler, n_sites: int, n_occ: int, rep: int):
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

        logger.debug(f"IQPE (n_sites={n_sites}, n_occ={n_occ}, repetition {rep}, iteration {num_iterations-k+1}) = {omega_coef}")

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


def analytic(model, n_sites, n_occ, model_params):
    H = model._build_H_matrix(n_sites, **model_params)
    eigvals, _ = np.linalg.eigh(H)
    kinetic_energy = np.sum(np.sort(eigvals)[:n_occ])

    mf_fn = getattr(model, 'mean_field_correction', None)
    interaction_energy = mf_fn(n_sites, n_occ, **model_params) if mf_fn else 0.0

    result = kinetic_energy + interaction_energy
    logger.info(f"Analytic (n_sites={n_sites}, n_occ={n_occ}) = {result}")
    return result


def vqe(n_sites, n_occ, model_params, fermionic_hamiltonian_fn, get_optimizer_fn, mapper, max_iters, n_layers, rep, backend=None):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(n_sites, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = EP_ansatz(n_sites, n_layers, n_occ)
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

        logger.debug(f"VQE (n_sites={n_sites}, n_occ={n_occ}, repetition {rep}) = {float(res.fun)}")
        return float(res.fun)


def iqpe(n_sites, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, rep, backend=None):
    np.seterr(all='ignore')

    fermionic_hamiltonian = fermionic_hamiltonian_fn(n_sites, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=time_param, synthesis=st)
    initial = HartreeFock(n_sites, (n_occ // 2 + n_occ % 2, n_occ // 2), mapper)

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

    phase, iteration_phases = iqpe_estimate(evolution, initial, n_iters, sampler, n_sites, n_occ, rep)
    res = float(-2 * np.pi * phase / time_param)
    iteration_energies = [float(-2 * np.pi * p / time_param) for p in iteration_phases]

    logger.debug(f"IQPE (n_sites={n_sites}, n_occ={n_occ}, repetition {rep}) = {res}")
    return res, iteration_energies


def vqe_other_benchmarks(n_sites, n_occ, model_params, fermionic_hamiltonian_fn, mapper, max_iters, n_layers, vqe_reps=1, backend=None):
    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = EP_ansatz(n_sites, n_layers, n_occ)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    fermionic_hamiltonian = fermionic_hamiltonian_fn(n_sites, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    num_queries = qubit_hamiltonian.size * max_iters * vqe_reps
    full_circuit_depth = ansatz_circuit.depth()
    two_gate_circuit_depth = ansatz_circuit.depth(lambda x: x.operation.num_qubits == 2)

    logger.info(f"VQE other benchmarks (n_sites={n_sites}, n_occ={n_occ}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth},{two_gate_circuit_depth}]")
    return num_queries, (full_circuit_depth, two_gate_circuit_depth)


def iqpe_other_benchmarks(n_sites, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, iqpe_reps, backend=None):
    np.seterr(all='ignore')
    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    fermionic_hamiltonian = fermionic_hamiltonian_fn(n_sites, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=time_param, synthesis=st)
    initial = HartreeFock(n_sites, (n_occ // 2 + n_occ % 2, n_occ // 2), mapper)

    full_circuit_depth = two_gate_circuit_depth = 0
    for k in range(n_iters, 0, -1):
        qc = construct_iqpe_circuit(evolution, initial, k, -2 * np.pi)
        qc = transpile(qc, backend=simulator, optimization_level=3)
        full_circuit_depth += qc.depth()
        two_gate_circuit_depth += qc.depth(lambda x: x.operation.num_qubits == 2)
        logger.debug(f"IQPE other benchmarks (n_sites={n_sites}, n_occ={n_occ}, iteration {n_iters-k+1}): circuit_depth=[{qc.depth(), qc.depth(lambda x: x.operation.num_qubits == 2)}]")

    num_queries = qubit_hamiltonian.size * iqpe_reps * n_trot * n_iters

    logger.info(f"IQPE other benchmarks (n_sites={n_sites}, n_occ={n_occ}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth // n_iters},{two_gate_circuit_depth // n_iters}]")
    return num_queries, (full_circuit_depth // n_iters, two_gate_circuit_depth // n_iters)
