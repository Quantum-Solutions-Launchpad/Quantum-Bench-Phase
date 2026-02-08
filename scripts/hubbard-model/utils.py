import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import excitation_preserving, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit_ibm_runtime import Session, Estimator

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit_nature.second_q.circuit.library import SlaterDeterminant, HartreeFock
from qiskit_nature.second_q.mappers import QubitMapper

from qiskit_algorithms.optimizers import SPSA

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit_aer.noise import NoiseModel
from qiskit.providers import BackendV2

import sys
from pathlib import Path
from loguru import logger

def setup_logging(debug_enabled: bool = True):
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt_console_info = "[<bold><green>{time:HH:mm:ss}</green></bold>] <white>{message}</white>"
    fmt_console_debug = "[<dim><white>{time:HH:mm:ss}</white></dim>] <dim>{message}</dim>"
    fmt_file = "[{time:HH:mm:ss}] {level}: {message}"

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

    logger.add(
        log_dir / "{time:YYYY-MM-DD}.log",
        level="DEBUG",
        colorize=False,
        format=fmt_file,
        rotation="00:00",
        retention="14 days",
        enqueue=True,
    )

    return logger

def real_space_slater_determinant(n_sites: int, t: float, U: float, n_occ: int) -> SlaterDeterminant:
    lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    spin = 2

    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)

    for i, j in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            H[s1, s2] -= t
            H[s2, s1] -= t

    quadratic_hamiltonian = QuadraticHamiltonian(H)
    transformation_matrix, _, _ = quadratic_hamiltonian.diagonalizing_bogoliubov_transform()

    occupied_orbitals = transformation_matrix[:n_occ, :]
    return SlaterDeterminant(occupied_orbitals)

def real_space_EP_ansatz(n_sites: int, n_layers: int, n_occ: int):
    spin = 2
    ansatz = QuantumCircuit(n_sites * spin)
    for i in range(n_occ):
        ansatz.x(i)
    ansatz.compose(excitation_preserving(n_sites * spin, "fsim", "linear", reps=n_layers), inplace=True)
    return ansatz

def iqpe_estimate(unitary: QuantumCircuit, state_preparation: QuantumCircuit, num_iterations: int, sampler: Sampler, n_sites: int, n_occ: int, rep: int):
    omega_coef = 0

    for k in range(num_iterations, 0, -1):
        omega_coef /= 2

        qc = construct_iqpe_circuit(unitary, state_preparation, k, -2*np.pi*omega_coef)

        sampler_job = sampler.run([qc])
        result = sampler_job.result().quasi_dists[0]
        x = 1 if result.get(1, 0) > result.get(0, 0) else 0

        omega_coef = omega_coef + x / 2

        logger.debug(f"IQPE (n_sites={n_sites}, n_occ={n_occ}, repetition {rep}, iteration {num_iterations-k+1}) = {omega_coef}")

    return omega_coef

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

def real_space_fermionic_hamiltonian(n_sites: int, t: float, U: float) -> FermionicOp:
    lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    spin = 2

    hamiltonian = 0.0 * FermionicOp({})

    for i, j in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            hamiltonian -= FermionicOp({
                f"+_{s1} -_{s2}": t,
                f"+_{s2} -_{s1}": t
            })

    for i in range(n_sites):
        spin_up = i * spin + 0
        spin_down = i * spin + 1
        hamiltonian += FermionicOp({
            f"+_{spin_up} -_{spin_up} +_{spin_down} -_{spin_down}": U
        })

    return hamiltonian

def real_space_exact(n_sites: int, t: float, U: float, n_occ: int) -> float:
    lattice = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    spin = 2

    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)
    for i, j in lattice:
        for s in range(spin):
            s1 = i * spin + s
            s2 = j * spin + s
            H[s1, s2] -= t
            H[s2, s1] -= t

    eigvals, _ = np.linalg.eigh(H)
    kinetic_energy = np.sum(np.sort(eigvals)[:n_occ])

    if U != 0:
        avg_double_occupancy = (n_occ / (2 * n_sites)) ** 2
        interaction_energy = U * n_sites * avg_double_occupancy
    else:
        interaction_energy = 0.0

    result = kinetic_energy + interaction_energy
    logger.info(f"Exact (n_sites={n_sites}, n_occ={n_occ}, t={t}, U={U}) = {result}")
    return result

def real_space_vqe(n_sites: int, t: float, U: float, n_occ: int, mapper: QubitMapper, max_iters: int, n_layers: int, rep: int, backend: BackendV2 = None):
    fermionic_hamiltonian = real_space_fermionic_hamiltonian(n_sites, t, U)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = real_space_EP_ansatz(n_sites, n_layers, n_occ) if n_layers else real_space_slater_determinant(n_sites, t, U, n_occ)
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

        spsa = SPSA(maxiter=max_iters)
        res = spsa.minimize(cost_func, x0=x0)

        logger.debug(f"VQE (n_sites={n_sites}, n_occ={n_occ}, t={t}, U={U}, repetition {rep}) = {float(res.fun)}")
        return float(res.fun)

def vqe_other_benchmarks(n_sites: int, t: float, U: float, n_occ: int, mapper: QubitMapper, max_iters: int, n_layers: int = 0, vqe_reps: int = 1, backend: BackendV2 = None) -> tuple[float, tuple[float]]:
    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    ansatz = real_space_EP_ansatz(n_sites, n_layers, n_occ) if n_layers else real_space_slater_determinant(n_sites, t, U, n_occ)
    ansatz_circuit = transpile(ansatz, backend=simulator, optimization_level=3)

    fermionic_hamiltonian = real_space_fermionic_hamiltonian(n_sites, t, U)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    num_queries = qubit_hamiltonian.size * max_iters * vqe_reps
    full_circuit_depth, two_gate_circuit_depth = ansatz_circuit.depth(), ansatz_circuit.depth(lambda x: x.operation.num_qubits == 2)

    logger.info(f"VQE other benchmarks (n_sites={n_sites}, n_occ={n_occ}, t={t}, U={U}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth},{two_gate_circuit_depth}]")
    return num_queries, (full_circuit_depth, two_gate_circuit_depth)

def real_space_iqpe(n_sites: int, t: float, U: float, n_occ: int, mapper: QubitMapper, time_param: float, n_trot: int, n_iters: int, rep: int, backend: BackendV2 = None) -> float:
    np.seterr(all='ignore')

    fermionic_hamiltonian = real_space_fermionic_hamiltonian(n_sites, t, U)
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

    phase = iqpe_estimate(evolution, initial, n_iters, sampler, n_sites, n_occ, rep)
    res = -2 * np.pi * phase / time_param

    logger.debug(f"IQPE (n_sites={n_sites}, n_occ={n_occ}, t={t}, U={U}, repetition {rep}) = {res}")
    return res

def iqpe_other_benchmarks(n_sites: int, t: float, U: float, n_occ: int, mapper: QubitMapper, time_param: float, n_trot: int, n_iters: int, iqpe_reps: int, backend: BackendV2 = None) -> tuple[float, tuple[float]]:
    np.seterr(all='ignore')
    if backend:
        noise_model = NoiseModel.from_backend(backend) if backend else NoiseModel()
        simulator = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    else:
        simulator = AerSimulator()

    fermionic_hamiltonian = real_space_fermionic_hamiltonian(n_sites, t, U)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=time_param, synthesis=st)
    initial = HartreeFock(n_sites, (n_occ // 2 + n_occ % 2, n_occ // 2), mapper)

    full_circuit_depth = two_gate_circuit_depth = 0
    for k in range(n_iters, 0, -1):
        qc = construct_iqpe_circuit(evolution, initial, k, -2*np.pi)
        qc = transpile(qc, backend=simulator, optimization_level=3)
        full_circuit_depth += qc.depth()
        two_gate_circuit_depth += qc.depth(lambda x: x.operation.num_qubits == 2)
        logger.debug(f"IQPE other benchmarks (n_sites={n_sites}, n_occ={n_occ}, iteration {n_iters-k+1}): circuit_depth=[{qc.depth(), qc.depth(lambda x: x.operation.num_qubits == 2)}]")

    num_queries = qubit_hamiltonian.size * iqpe_reps * n_trot * n_iters

    logger.info(f"IQPE other benchmarks (n_sites={n_sites}, n_occ={n_occ}, t={t}, U={U}): num_queries={num_queries}, circuit_depth=[{full_circuit_depth // n_iters},{two_gate_circuit_depth // n_iters}]")
    return num_queries, (full_circuit_depth // n_iters, two_gate_circuit_depth // n_iters)
