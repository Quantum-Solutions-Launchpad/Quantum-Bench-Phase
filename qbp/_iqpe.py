"""Iterative Quantum Phase Estimation (IQPE) simulation method and core solvers."""

from __future__ import annotations

import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp

from qbp._backend import make_iqpe_sampler
from qbp._core import (
    _fmt_params, _hf_initial_state, _make_simulator, _uniform_initial, run_reps_parallel, logger,
)
from qbp._mitigation import chain_correct_counts
from qbp._method import Method, ParamSpec, SimulationMethod, register_method


# --------------------------------------------------------------------- solvers

def _exact_bloch_unitary(H_matrix: np.ndarray, time_param: float) -> QuantumCircuit:
    """Build the EXACT time-evolution unitary e^{-iH*t} for a 2x2 Bloch Hamiltonian.

    For H = d0*I + dx*X + dy*Y + dz*Z, the exact unitary is:
        U = e^{-id0*t} * (cos(|d|*t)*I - i*sin(|d|*t)*(dx*X + dy*Y + dz*Z)/|d|)

    This is a global phase times a single-qubit rotation — no Trotter error.
    Works for any 2x2 Hermitian matrix (Bloch Hamiltonians are always 2x2).
    """
    assert H_matrix.shape == (2, 2), "Exact unitary only valid for 2x2 Bloch Hamiltonians"

    d0 = 0.5 * np.real(H_matrix[0, 0] + H_matrix[1, 1])
    dx = np.real(H_matrix[0, 1] + H_matrix[1, 0]) / 2
    dy = np.imag(H_matrix[1, 0] - H_matrix[0, 1]) / 2
    dz = np.real(H_matrix[0, 0] - H_matrix[1, 1]) / 2

    d_norm = np.sqrt(dx**2 + dy**2 + dz**2)

    phase = np.exp(-1j * d0 * time_param)
    if d_norm < 1e-10:
        U_matrix = phase * np.eye(2, dtype=complex)
    else:
        c = np.cos(d_norm * time_param)
        s = np.sin(d_norm * time_param)
        U_matrix = phase * np.array([
            [c - 1j * s * dz / d_norm,
             -1j * s * (dx - 1j * dy) / d_norm],
            [-1j * s * (dx + 1j * dy) / d_norm,
             c + 1j * s * dz / d_norm]
        ])

    qc = QuantumCircuit(1)
    qc.unitary(U_matrix, [0])
    return qc


def _bloch_eigenstate(H_matrix: np.ndarray, which: str = "min") -> QuantumCircuit:
    """Prepare the eigenstate of a 2x2 Bloch Hamiltonian for phase estimation.

    Classical-feedback IQPE only converges to a single well-defined phase
    when the input state is close to an eigenstate of the evolution
    unitary. A k-independent |+> state is only aligned with the eigenbasis
    when dz dominates dx/dy.
    """
    dx = np.real(H_matrix[0, 1] + H_matrix[1, 0]) / 2
    dy = np.imag(H_matrix[1, 0] - H_matrix[0, 1]) / 2
    dz = np.real(H_matrix[0, 0] - H_matrix[1, 1]) / 2
    d_norm = np.sqrt(dx**2 + dy**2 + dz**2)

    qc = QuantumCircuit(1)
    if d_norm < 1e-10:
        return qc  # degenerate point: any state is an eigenstate

    theta = np.arccos(np.clip(dz / d_norm, -1.0, 1.0))
    phi = np.arctan2(dy, dx)

    qc.ry(theta + np.pi if which == "min" else theta, 0)
    qc.rz(phi, 0)
    return qc


def iqpe_estimate(unitary: QuantumCircuit, state_preparation: QuantumCircuit, num_iterations: int, sampler, label: str = "", strategies=None):
    strategies = strategies or []
    omega_coef = 0
    iteration_phases = []

    for k in range(num_iterations, 0, -1):
        omega_coef /= 2

        qc = construct_iqpe_circuit(unitary, state_preparation, k, -2 * np.pi * omega_coef)

        if strategies and hasattr(sampler, "raw_dist"):
            raw_dist = sampler.raw_dist(qc)
            # The phase/ancilla register is added after the eigenstate
            # register in construct_iqpe_circuit, so it occupies physical
            # qubit index unitary.num_qubits, not 0 -- M3 needs the actual
            # measured qubit(s) to look up the right calibration, or it
            # would silently miscalibrate under a non-uniform noise model.
            result = chain_correct_counts(strategies, raw_dist, [unitary.num_qubits], qc.num_clbits)
            x = 1 if result.get(1, 0) > result.get(0, 0) else 0
        else:
            # Real-hardware samplers (Runtime/IQM) don't expose raw_dist --
            # mitigation was never wired up for real backends in this
            # project, so fall back to the sampler's own bit decision.
            x = sampler.sample_bit(qc)

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


def _iqpe_sparse(hamiltonian, initial, time_param, n_trot, n_iters, rep, backend=None, label="", strategies=None):
    np.seterr(all='ignore')
    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(hamiltonian, time=time_param, synthesis=st)

    with make_iqpe_sampler(backend) as sampler:
        phase, iteration_phases = iqpe_estimate(evolution, initial, n_iters, sampler, label, strategies=strategies)
    res = float(-2 * np.pi * phase / time_param)
    iter_energies = [float(-2 * np.pi * p / time_param) for p in iteration_phases]
    logger.debug(f"IQPE {label} = {res}")
    return res, iter_energies


def _iqpe_exact(unitary_qc, initial, time_param, n_iters, rep, backend=None, label="", strategies=None):
    """Run IQPE with an exact pre-built unitary circuit (no Trotterization)."""
    np.seterr(all='ignore')
    with make_iqpe_sampler(backend) as sampler:
        phase, iteration_phases = iqpe_estimate(unitary_qc, initial, n_iters, sampler, label, strategies=strategies)
    res = float(-2 * np.pi * phase / time_param)
    iter_energies = [float(-2 * np.pi * p / time_param) for p in iteration_phases]
    logger.debug(f"IQPE {label} = {res}")
    return res, iter_energies


def iqpe_fermionic(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, rep, backend=None, get_initial_state_fn=None, strategies=None, initial_state_override=None):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    if initial_state_override is not None:
        # Warm-started state (a converged VQE ansatz) that has real
        # overlap with the ground state. A bare HF reference can have
        # near-zero overlap for correlated/near-degenerate Hamiltonians,
        # which makes classical-feedback IQPE converge to a spurious phase
        # regardless of noise or mitigation.
        initial = initial_state_override
    elif get_initial_state_fn is not None:
        initial = get_initial_state_fn(qubit_hamiltonian, n_occ=n_occ, spin=spin, mapper=mapper)
    else:
        initial = _hf_initial_state(n_sites, spin, n_occ, mapper)
    label = f"({_fmt_params(lattice, n_occ, model_params, repetition=rep)})"
    return _iqpe_sparse(qubit_hamiltonian, initial, time_param, n_trot, n_iters, rep, backend=backend, label=label, strategies=strategies)


def iqpe_observable(model, lattice, n_sites, spin, n_occ, model_params, mapper, time_param, n_trot, n_iters, rep, observable, backend=None, get_initial_state_fn=None, fermionic_hamiltonian_fn=None):
    obs = model.get_observable(observable)
    fermionic_hamiltonian_fn = fermionic_hamiltonian_fn or model.fermionic_hamiltonian

    def sub_eval(sub_n_occ, observable_qubit_ops=None):
        if observable_qubit_ops is not None:
            raise NotImplementedError(
                "IQPE only supports observables whose composite uses energies; "
                "operator-measurement observables require VQE."
            )
        energy, _ = iqpe_fermionic(
            lattice, n_sites, spin, sub_n_occ, model_params,
            fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, rep,
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
            fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, rep,
            backend=backend, get_initial_state_fn=get_initial_state_fn,
        )
    raise NotImplementedError(
        f"IQPE cannot directly measure observable '{observable}' "
        f"(requires operator measurement on the prepared state). Use VQE instead."
    )


def iqpe_bloch(k_tuple, model_params, bloch_hamiltonian_fn, time_param, n_trot, n_iters, rep, backend=None, strategies=None, extremum="min"):
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    label = f"bloch (k={tuple(round(float(x), 3) for x in k_tuple)}, rep={rep})"

    if H_matrix.shape == (2, 2):
        exact_unitary = _exact_bloch_unitary(H_matrix, time_param)
        initial = _bloch_eigenstate(H_matrix, which=extremum)
        return _iqpe_exact(exact_unitary, initial, time_param, n_iters, rep,
                           backend=backend, label=label, strategies=strategies)
    else:
        hamiltonian = SparsePauliOp.from_operator(H_matrix)
        initial = _uniform_initial(hamiltonian.num_qubits)
        return _iqpe_sparse(hamiltonian, initial, time_param, n_trot, n_iters, rep,
                            backend=backend, label=label, strategies=strategies)


def iqpe_operator(hamiltonian, time_param, n_trot, n_iters, rep, extremum="min", backend=None, label="", get_initial_state_fn=None, observable="E"):
    op = hamiltonian * -1 if extremum == "max" else hamiltonian
    if get_initial_state_fn is not None:
        initial = get_initial_state_fn(op)
    else:
        initial = _uniform_initial(hamiltonian.num_qubits)
    iqpe_label = f"[{observable}] ({label}, rep={rep})" if label else f"[{observable}] (rep={rep})"
    energy, iter_energies = _iqpe_sparse(op, initial, time_param, n_trot, n_iters, rep, backend=backend, label=iqpe_label)
    if extremum == "max":
        energy = -energy
        iter_energies = [-e for e in iter_energies]
    return energy, iter_energies


def iqpe_other_benchmarks(lattice, n_sites, spin, n_occ, model_params, fermionic_hamiltonian_fn, mapper, time_param, n_trot, n_iters, iqpe_reps, backend=None):
    np.seterr(all='ignore')
    simulator = _make_simulator(backend)

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
    simulator = _make_simulator(backend)

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


def _iqpe_iteration_seconds(unitary, initial, n_iters, backend, shots):
    from qbp._backend import circuit_qpu_seconds

    total = 0.0
    for k in range(n_iters, 0, -1):
        qc = construct_iqpe_circuit(unitary, initial, k, -2 * np.pi)
        total += circuit_qpu_seconds(backend, qc, shots)
    return total


def iqpe_cell_seconds(lattice, n_sites, spin, n_occ, model_params,
                      fermionic_hamiltonian_fn, mapper, time_param, n_trot,
                      n_iters, reps, backend, shots):
    fermionic_hamiltonian = fermionic_hamiltonian_fn(lattice, **model_params)
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(qubit_hamiltonian, time=time_param, synthesis=st)
    initial = _hf_initial_state(n_sites, spin, n_occ, mapper)
    return _iqpe_iteration_seconds(evolution, initial, n_iters, backend, shots) * reps


def iqpe_bloch_seconds(k_tuple, model_params, bloch_hamiltonian_fn, time_param,
                       n_trot, n_iters, reps, backend, shots):
    H_matrix = bloch_hamiltonian_fn(*k_tuple, **model_params)
    if H_matrix.shape == (2, 2):
        unitary = _exact_bloch_unitary(H_matrix, time_param)
        initial = _bloch_eigenstate(H_matrix, which="min")
    else:
        hamiltonian = SparsePauliOp.from_operator(H_matrix)
        st = SuzukiTrotter(reps=n_trot)
        unitary = PauliEvolutionGate(hamiltonian, time=time_param, synthesis=st)
        initial = _uniform_initial(hamiltonian.num_qubits)
    return _iqpe_iteration_seconds(unitary, initial, n_iters, backend, shots) * reps


def iqpe_operator_seconds(hamiltonian, time_param, n_trot, n_iters, reps,
                          get_initial_state_fn, backend, shots):
    st = SuzukiTrotter(reps=n_trot)
    evolution = PauliEvolutionGate(hamiltonian, time=time_param, synthesis=st)
    if get_initial_state_fn is not None:
        initial = get_initial_state_fn(hamiltonian)
    else:
        initial = _uniform_initial(hamiltonian.num_qubits)
    return _iqpe_iteration_seconds(evolution, initial, n_iters, backend, shots) * reps


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
        ParamSpec("warm_start_vqe", bool, False,
                   "Warm-start the real-space initial state with a converged VQE "
                   "ansatz instead of bare Hartree-Fock. HF can have near-zero "
                   "overlap with correlated/near-degenerate ground states, which "
                   "makes classical-feedback IQPE converge to a spurious phase "
                   "regardless of noise or mitigation.", is_flag=True),
        ParamSpec("warm_start_iters", int, 200, "Optimizer iterations for the warm-start VQE", metavar="N"),
        ParamSpec("warm_start_layers", int, 4, "Ansatz layers for the warm-start VQE", metavar="N"),
    ]
    EXTRA_PARAMS = ("initial_state",)
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = True
    SUPPORTS_OPERATOR = True

    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        ferm_fn = ctx.fermionic_hamiltonian_fn or model.fermionic_hamiltonian
        warm_start_circuit = None
        if self.warm_start_vqe and observable == "E":
            from qbp._vqe import vqe_fermionic
            _, warm_start_circuit = vqe_fermionic(
                lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                model.fermionic_hamiltonian, model.get_optimizer,
                model.get_vqe_ansatz, ctx.mapper,
                self.warm_start_iters, self.warm_start_layers, rep=0,
                backend=backend, return_state=True, seed=ctx.cell_index,
            )

        def run_one_rep(rep):
            if observable == "E":
                energy, iter_energies = iqpe_fermionic(
                    lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ferm_fn, ctx.mapper,
                    self.time, self.trot, self.iters, rep,
                    backend=backend, get_initial_state_fn=model.get_iqpe_initial_state,
                    strategies=self.mitigation_strategies, initial_state_override=warm_start_circuit,
                )
            else:
                energy, iter_energies = iqpe_observable(
                    model, lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
                    ctx.mapper, self.time, self.trot, self.iters, rep, observable,
                    backend=backend, get_initial_state_fn=model.get_iqpe_initial_state,
                    fermionic_hamiltonian_fn=ferm_fn,
                )
            return (float(energy), iter_energies)

        rep_results = run_reps_parallel(self.reps, run_one_rep)
        reps = [energy for energy, _ in rep_results]
        iteration_energies = [iter_energies for _, iter_energies in rep_results]
        num_queries, (total, two_q) = iqpe_other_benchmarks(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            ferm_fn, ctx.mapper,
            self.time, self.trot, self.iters, self.reps, backend=backend,
        )
        return {
            "repetitions": reps,
            "iteration_energies": iteration_energies,
            "num_queries": num_queries,
            "circuit_depth": {"total": total, "two_qubit": two_q},
        }

    def compute_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                           backend, ctx):
        def run_one_rep(rep):
            energy, iter_energies = iqpe_bloch(
                k_tuple, cell_params, model.bloch_hamiltonian,
                self.time, self.trot, self.iters, rep, backend=backend,
                strategies=self.mitigation_strategies, extremum="min",
            )
            return (float(energy), iter_energies)

        rep_results = run_reps_parallel(self.reps, run_one_rep)
        reps = [energy for energy, _ in rep_results]
        iteration_energies = [iter_energies for _, iter_energies in rep_results]
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

    def compute_operator_cell(self, op, *, extremum, backend, label, observable: str = "E"):
        from qbp._yaml_model import InitialStateSpec, build_initial_state_factory
        spec = (
            InitialStateSpec.model_validate(self.initial_state) if self.initial_state
            else InitialStateSpec(type="uniform")
        )
        get_initial_state = build_initial_state_factory(spec, name="operator")

        def run_one_rep(rep):
            energy, iter_energies = iqpe_operator(
                op, self.time, self.trot, self.iters, rep, extremum, backend, label,
                get_initial_state, observable,
            )
            return (float(energy), iter_energies)

        rep_results = run_reps_parallel(self.reps, run_one_rep)
        reps = [energy for energy, _ in rep_results]
        iteration_energies = [iter_energies for _, iter_energies in rep_results]
        return {"repetitions": reps, "iteration_energies": iteration_energies}

    def _estimate_shots(self, backend, shots):
        from qbp._backend import _IQM_SHOTS, _IBM_SAMPLER_SHOTS, is_iqm_backend

        if shots is not None:
            return shots
        return _IQM_SHOTS if is_iqm_backend(backend) else _IBM_SAMPLER_SHOTS

    def estimate_cell(self, model, lattice, n_occ, cell_params, observable, *,
                      backend, ctx, shots):
        ferm_fn = ctx.fermionic_hamiltonian_fn or model.fermionic_hamiltonian
        return iqpe_cell_seconds(
            lattice, ctx.n_sites, ctx.spin, n_occ, cell_params,
            ferm_fn, ctx.mapper, self.time, self.trot, self.iters, self.reps,
            backend, self._estimate_shots(backend, shots),
        )

    def estimate_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                            backend, ctx, shots):
        return iqpe_bloch_seconds(
            k_tuple, cell_params, model.bloch_hamiltonian,
            self.time, self.trot, self.iters, self.reps, backend,
            self._estimate_shots(backend, shots),
        )

    def estimate_operator_cell(self, op, *, extremum, backend, observable="E", shots):
        from qbp._yaml_model import InitialStateSpec, build_initial_state_factory

        spec = (
            InitialStateSpec.model_validate(self.initial_state) if self.initial_state
            else InitialStateSpec(type="uniform")
        )
        get_initial_state = build_initial_state_factory(spec, name="operator")
        return iqpe_operator_seconds(
            op, self.time, self.trot, self.iters, self.reps,
            get_initial_state, backend, self._estimate_shots(backend, shots),
        )