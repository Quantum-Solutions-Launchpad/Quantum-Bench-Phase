"""Iterative Quantum Phase Estimation (IQPE) simulation method and core solvers."""

from __future__ import annotations

import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from quaph._core import (
    _fmt_params, _hf_initial_state, _make_sampler, _uniform_initial, logger,
)
from quaph._method import Method, ParamSpec, SimulationMethod, register_method


# --------------------------------------------------------------------- solvers
def iqpe_estimate(unitary: QuantumCircuit, state_preparation: QuantumCircuit, num_iterations: int, sampler: Sampler, label: str = ""):
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

        logger.debug(f"IQPE {label} iteration={num_iterations-k+1} = {omega_coef}")

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


def _iqpe_sparse(hamiltonian, initial, time_param, n_trot, n_iters, rep, backend=None, label=""):
    np.seterr(all='ignore')
    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(hamiltonian, time=time_param, synthesis=st)
    sampler = _make_sampler(backend)

    phase, iteration_phases = iqpe_estimate(evolution, initial, n_iters, sampler, label)
    res = float(-2 * np.pi * phase / time_param)
    iter_energies = [float(-2 * np.pi * p / time_param) for p in iteration_phases]
    logger.debug(f"IQPE {label} = {res}")
    return res, iter_energies


def iqpe_fermionic(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, rep, backend=None, get_initial_state_fn=None):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    if get_initial_state_fn is not None:
        initial = get_initial_state_fn(qubit_hamiltonian, n_occ=n_occ, spin=spin, mapper=mapper)
    else:
        initial = _hf_initial_state(n_sites, spin, n_occ, mapper)
    label = f"({_fmt_params(lattice, n_occ, model_params, repetition=rep)})"
    return _iqpe_sparse(qubit_hamiltonian, initial, time_param, n_trot, n_iters, rep, backend=backend, label=label)


def iqpe_observable(model, lattice, n_sites, spin, n_occ, model_params, mapper, time_param, n_trot, n_iters, rep, observable, backend=None, get_initial_state_fn=None):
    obs = model.get_observable(observable)

    def sub_eval(sub_n_occ, observable_qubit_ops=None):
        if observable_qubit_ops is not None:
            raise NotImplementedError(
                "IQPE only supports observables whose composite uses energies; "
                "operator-measurement observables require VQE."
            )
        energy, _ = iqpe_fermionic(
            lattice, n_sites, spin, sub_n_occ, model_params,
            model.fermionic_hamiltonian, mapper, time_param, n_trot, n_iters, rep,
            backend=backend, get_initial_state_fn=get_initial_state_fn,
        )
        return energy

    if obs.quantum_composite is not None:
        n_orbitals = n_sites * spin
        return float(obs.quantum_composite(
            model, lattice, n_occ, model_params, mapper, n_orbitals, sub_eval,
        )), []
    if observable == "E":
        return iqpe_fermionic(
            lattice, n_sites, spin, n_occ, model_params,
            model.fermionic_hamiltonian, mapper, time_param, n_trot, n_iters, rep,
            backend=backend, get_initial_state_fn=get_initial_state_fn,
        )
    raise NotImplementedError(
        f"IQPE cannot directly measure observable '{observable}' "
        f"(requires operator measurement on the prepared state). Use VQE instead."
    )


def iqpe_bloch(k_tuple, model_params, bloch_hamiltonian_fn, time_param, n_trot, n_iters, rep, backend=None):
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    hamiltonian = SparsePauliOp.from_operator(H_matrix)
    initial = _uniform_initial(hamiltonian.num_qubits)
    label = f"bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep})"
    return _iqpe_sparse(hamiltonian, initial, time_param, n_trot, n_iters, rep, backend=backend, label=label)


def iqpe_operator(hamiltonian, time_param, n_trot, n_iters, rep, extremum="min", backend=None, label="", get_initial_state_fn=None):
    op = hamiltonian * -1 if extremum == "max" else hamiltonian
    if get_initial_state_fn is not None:
        initial = get_initial_state_fn(op)
    else:
        initial = _uniform_initial(hamiltonian.num_qubits)
    iqpe_label = f"[operator] ({label}, rep={rep})" if label else f"[operator] (rep={rep})"
    energy, iter_energies = _iqpe_sparse(op, initial, time_param, n_trot, n_iters, rep, backend=backend, label=iqpe_label)
    if extremum == "max":
        energy = -energy
        iter_energies = [-e for e in iter_energies]
    return energy, iter_energies


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


def iqpe_supports_observable(model, observable: str) -> bool:
    """IQPE can only target energy or energy-composite observables (e.g. charge_gap)."""
    if observable == "E":
        return True
    obs = model.get_observable(observable)
    return obs.quantum_composite is not None and observable == "charge_gap"


# ---------------------------------------------------------------------- method
@register_method
class IQPEMethod(SimulationMethod):
    METHOD = Method.IQPE
    LABEL = "IQPE"
    PARAM_SPECS = [
        ParamSpec("time", float, 0.1, "Hamiltonian evolution time", metavar="F"),
        ParamSpec("trot", int, 1, "Suzuki-Trotter steps", metavar="N"),
        ParamSpec("iters", int, 1, "IQPE phase-estimation iterations", metavar="N"),
        ParamSpec("reps", int, 1, "Number of independent IQPE repetitions", metavar="N"),
    ]
    # Dict-valued param used only on the --qubit-operator path.
    EXTRA_PARAMS = ("initial_state",)
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = True
    SUPPORTS_OPERATOR = True

    # ----------------------------------------------------------------- real space
    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        reps = []
        iteration_energies = []
        for rep in range(1, self.reps + 1):
            if observable == "E":
                energy, iter_energies = iqpe_fermionic(
                    lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    model.fermionic_hamiltonian, ctx.mapper,
                    self.time, self.trot, self.iters, rep,
                    backend=backend, get_initial_state_fn=model.get_iqpe_initial_state,
                )
            else:
                energy, iter_energies = iqpe_observable(
                    model, lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ctx.mapper, self.time, self.trot, self.iters, rep, observable,
                    backend=backend, get_initial_state_fn=model.get_iqpe_initial_state,
                )
            reps.append(float(energy))
            iteration_energies.append(iter_energies)
        num_queries, (total, two_q) = iqpe_other_benchmarks(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            model.fermionic_hamiltonian, ctx.mapper,
            self.time, self.trot, self.iters, self.reps, backend=backend,
        )
        return {
            "repetitions": reps,
            "iteration_energies": iteration_energies,
            "num_queries": num_queries,
            "circuit_depth": {"total": total, "two_qubit": two_q},
        }

    # -------------------------------------------------------------- band structure
    def compute_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                           backend, ctx):
        reps = []
        iteration_energies = []
        for rep in range(1, self.reps + 1):
            energy, iter_energies = iqpe_bloch(
                k_tuple, cell_params, model.bloch_hamiltonian,
                self.time, self.trot, self.iters, rep, backend=backend,
            )
            reps.append(float(energy))
            iteration_energies.append(iter_energies)
        num_queries, (total, two_q) = iqpe_bloch_other_benchmarks(
            k_tuple, cell_params, model.bloch_hamiltonian,
            self.time, self.trot, self.iters, self.reps, backend=backend,
        )
        return {
            "repetitions": reps,
            "iteration_energies": iteration_energies,
            "num_queries": num_queries,
            "circuit_depth": {"total": total, "two_qubit": two_q},
        }

    # ------------------------------------------------------------------- operator
    def compute_operator_cell(self, op, *, extremum, backend, label):
        from quaph._yaml_model import InitialStateSpec, build_initial_state_factory
        spec = (
            InitialStateSpec.model_validate(self.initial_state) if self.initial_state
            else InitialStateSpec(type="uniform")
        )
        get_initial_state = build_initial_state_factory(spec, name="operator")
        reps = []
        iteration_energies = []
        for rep in range(1, self.reps + 1):
            energy, iter_energies = iqpe_operator(
                op, self.time, self.trot, self.iters, rep, extremum, backend, label,
                get_initial_state,
            )
            reps.append(float(energy))
            iteration_energies.append(iter_energies)
        return {"repetitions": reps, "iteration_energies": iteration_energies}
