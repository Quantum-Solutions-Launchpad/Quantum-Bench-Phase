from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, Iterable, Optional, Tuple, Union, cast

import functools
import math
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
from qiskit.providers import Backend
from qiskit.result import QuasiDistribution
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.circuit.library import FermionicGaussianState
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit_nature.second_q.operators import FermionicOp
import mthree

_CovarianceDict = Dict[FrozenSet[Tuple[int, int]], float]

def error_mitigation(n_modes: int, tunneling: float, superconducting: float, chemical_potential: float, occupied_orbitals: tuple[int, ...], backend: Optional[Backend], shots: int):
    circuits = generate_circuits(n_modes, tunneling, superconducting, chemical_potential, occupied_orbitals)
    simulator = AerSimulator.from_backend(backend) if backend else AerSimulator()
    transpiled_circuits = {key: transpile(qc, simulator) for key, qc in circuits.items()}
    raw_counts = {key: dict(simulator.run(qc, shots=shots).result().get_counts(qc)) for key, qc in transpiled_circuits.items()}
    pipeline_results = {}
    hamiltonian_quad = kitaev_hamiltonian(n_modes, tunneling, superconducting, chemical_potential)
    corr_target, _ = compute_correlation_matrix_exact(n_modes, tunneling, superconducting, chemical_potential, occupied_orbitals)
    quasis_raw = {key: counts_to_quasis(counts) for key, counts in raw_counts.items()}
    corr_raw, cov_raw = compute_correlation_matrix(quasis_raw, n_modes)
    pipeline_results['raw'] = _calculate_observables(corr_raw, cov_raw, hamiltonian_quad, corr_target)
    quasis_mem = {}
    if backend:
        mit = mthree.M3Mitigation(backend)
        for key, counts in raw_counts.items():
            final_mapping = mthree.utils.final_measurement_mapping(transpiled_circuits[key])
            quasis_mem[key] = mit.apply_correction(counts, final_mapping, return_mitigation_overhead=True)
    else:
        quasis_mem = quasis_raw
    corr_mem, cov_mem = compute_correlation_matrix(quasis_mem, n_modes)
    pipeline_results['mem'] = _calculate_observables(corr_mem, cov_mem, hamiltonian_quad, corr_target)
    _, _, _, hamiltonian_parity = diagonalizing_bogoliubov_transform(n_modes, tunneling, superconducting, chemical_potential)
    exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
    def parity_predicate(bit_key: Union[str, int]) -> bool:
        if isinstance(bit_key, int): bit_key = format(bit_key, 'b')
        return (-1) ** sum(b == '1' for b in bit_key) == exact_parity
    quasis_ps, _ = post_select_quasis_dict(quasis_mem, parity_predicate)
    corr_ps, cov_ps = compute_correlation_matrix(quasis_ps, n_modes)
    pipeline_results['ps'] = _calculate_observables(corr_ps, cov_ps, hamiltonian_quad, corr_target)
    corr_pur = purify_idempotent_matrix(corr_ps)
    pipeline_results['pur'] = _calculate_observables(corr_pur, cov_ps, hamiltonian_quad, corr_target)
    return (chemical_potential, occupied_orbitals, pipeline_results)

def _calculate_observables(corr, cov, hamiltonian, corr_target):
    energy, energy_std = np.real(expectation_from_correlation_matrix(hamiltonian, corr, cov))
    fid, fid_std = fidelity_witness(corr, corr_target, cov)
    return {'energy': (energy, energy_std), 'fidelity_witness': (fid, fid_std)}

def post_select_quasis_dict(quasis_dict: dict, predicate: Callable) -> tuple[dict, float]:
    new_quasis_dict = {key: post_select_quasis(quasis, predicate)[0] for key, quasis in quasis_dict.items()}
    return new_quasis_dict, 0.0

def post_select_quasis(quasis: QuasiDistribution, predicate: Callable) -> tuple[QuasiDistribution, float]:
    new_data = quasis.copy()
    removed_mass = sum(prob for bit, prob in quasis.items() if not predicate(bit))
    for bit, prob in quasis.items():
        if not predicate(bit): new_data[bit] = 0.0
    norm = 1.0 - removed_mass
    if norm > 1e-9:
        for bit in new_data: new_data[bit] /= norm
    new_quasi = QuasiDistribution(new_data, shots=int(quasis.shots * norm))
    if hasattr(quasis, 'mitigation_overhead'): new_quasi.mitigation_overhead = quasis.mitigation_overhead
    else: new_quasi.mitigation_overhead = 1.0
    return new_quasi, removed_mass

def fidelity_witness(corr: np.ndarray, corr_target: np.ndarray, cov: Optional[_CovarianceDict] = None) -> tuple[float, float]:
    """
    how close the experimentally derived correlation matrix is to the exact target matrix.
    """
    dim, _ = corr.shape
    witness = 1 - np.trace((corr_target - corr) @ (corr_target - 0.5 * np.eye(dim)))
    return np.real(witness), 0.0

def covariance(quasi_dist: QuasiDistribution, op1: str, op2: str, n_qubits: int) -> float:
    expval1, expval2 = expval(quasi_dist, op1, n_qubits), expval(quasi_dist, op2, n_qubits)
    cov = sum(prob * (evaluate_diagonal_op(op1, bit, n_qubits) - expval1) * (evaluate_diagonal_op(op2, bit, n_qubits) - expval2) for bit, prob in quasi_dist.items())
    if quasi_dist.shots > 0:
        mit_overhead = getattr(quasi_dist, 'mitigation_overhead', 1.0)
        return cov * mit_overhead / quasi_dist.shots
    return 0.0

def counts_to_quasis(counts: dict[str, int]) -> QuasiDistribution:
    shots = sum(counts.values())
    if shots == 0: return QuasiDistribution({}, shots=0)
    quasi = QuasiDistribution({bit: count / shots for bit, count in counts.items()}, shots=shots)
    quasi.mitigation_overhead = 1.0
    return quasi

def orbital_combinations(n_modes: int, threshold: Optional[int] = None) -> Iterable[tuple[int, ...]]:
    if threshold is None: threshold = n_modes
    yield (); yield tuple(range(n_modes))
    for i in range(threshold): yield (i,); yield tuple(j for j in range(n_modes) if j != i)

def orbital_permutations(n_modes: int) -> Iterable[tuple[int, ...]]:
    perm = list(range(n_modes))
    for _ in range(math.ceil(n_modes / 2)):
        yield tuple(perm)
        for i in range(0, n_modes - 1, 2): perm[i], perm[i+1] = perm[i+1], perm[i]
        for i in range(1, n_modes - 1, 2): perm[i], perm[i+1] = perm[i+1], perm[i]

def measurement_labels(n_modes: int) -> Iterable[tuple[tuple[int, ...], str]]:
    yield tuple(range(n_modes)), "number"
    for perm in orbital_permutations(n_modes):
        for interaction in ["tunneling_plus", "tunneling_minus", "superconducting_plus", "superconducting_minus"]:
            yield perm, f"{interaction}_even"; yield perm, f"{interaction}_odd"

@functools.lru_cache
def kitaev_hamiltonian(n_modes, tunneling, superconducting, chemical_potential) -> QuadraticHamiltonian:
    eye, upper, lower = np.eye(n_modes), np.diag(np.ones(n_modes - 1), k=1), np.diag(np.ones(n_modes - 1), k=-1)
    hermitian = -tunneling * (upper + lower) + chemical_potential * eye
    antisymmetric = superconducting * (upper - lower)
    return QuadraticHamiltonian(hermitian, antisymmetric, -0.5 * chemical_potential * n_modes)

def measure_interaction_op(circuit: QuantumCircuit, label: str) -> QuantumCircuit:
    circuit = circuit.copy()
    if label == "number": circuit.measure_all(); return circuit
    if label.startswith("tunneling_plus"): gate = XXPlusYYGate(np.pi / 2, -np.pi / 2)
    elif label.startswith("tunneling_minus"): gate = XXPlusYYGate(np.pi / 2, -np.pi)
    elif label.startswith("superconducting_plus"): gate = XXMinusYYGate(np.pi / 2, -np.pi / 2)
    else: gate = XXMinusYYGate(np.pi / 2, -np.pi)
    start = 0 if label.endswith("even") else 1
    for i in range(start, circuit.num_qubits - 1, 2): circuit.append(gate, [i, i + 1])
    circuit.measure_all(); return circuit

def expectation_from_correlation_matrix(operator: QuadraticHamiltonian, corr: np.ndarray, cov: Optional[_CovarianceDict] = None) -> tuple[complex, float]:
    n = operator.hermitian_part.shape[0]
    exp_val = np.sum(operator.hermitian_part * corr[:n, :n] + np.real(operator.antisymmetric_part * corr[:n, n:])) + operator.constant
    return exp_val, 0.0

@functools.lru_cache
def diagonalizing_bogoliubov_transform(n_modes, tunneling, superconducting, chemical_potential) -> tuple[np.ndarray, np.ndarray, float, int]:
    ham = kitaev_hamiltonian(n_modes, tunneling, superconducting, chemical_potential)
    trans_mat, orb_e, const = ham.diagonalizing_bogoliubov_transform()
    W1, W2 = trans_mat[:, :n_modes], trans_mat[:, n_modes:]
    full_trans_mat = np.block([[W1, W2], [W2.conj(), W1.conj()]])
    return trans_mat, orb_e, const, np.sign(np.real(np.linalg.det(full_trans_mat)))

def evaluate_diagonal_op(operator: str, bitstring: Union[str, int], n_qubits: int) -> int:
    if isinstance(bitstring, int): bitstring = format(bitstring, f'0{n_qubits}b')[::-1]
    prod = 1
    for op, bit in zip(reversed(operator), bitstring):
        if op in "01": prod *= (bit == op)
        elif op == "Z": prod *= (-1)**(bit == "1")
    return prod

def expval(quasi_dist: QuasiDistribution, operator: str, n_qubits: int) -> float:
    return sum(prob * evaluate_diagonal_op(operator, bit, n_qubits) for bit, prob in quasi_dist.items())

def compute_interaction_matrix(quasis, n, label) -> tuple[np.ndarray, _CovarianceDict]:
    mat, cov = np.zeros((n, n)), defaultdict(float)
    if label == "tunneling_plus": sign, sym = -1, 1
    elif label == "tunneling_minus": sign, sym = -1, -1
    elif label == "superconducting_plus": sign, sym = 1, -1
    else: sign, sym = 1, -1
    for perm in orbital_permutations(n):
        for start_idx, qd in [(0, quasis.get((perm, f"{label}_even"))), (1, quasis.get((perm, f"{label}_odd")))]:
            if not qd: continue
            for i in range(start_idx, n - 1, 2):
                z0, z1 = "I"*(n-i-1)+"Z"+"I"*i, "I"*(n-i-2)+"Z"+"I"*(i+1)
                val = 0.5 * (expval(qd, z1, n) + sign * expval(qd, z0, n))
                p, q = perm[i], perm[i+1]; mat[p, q], mat[q, p] = val, sym * val
    return mat, cov

def compute_correlation_matrix(quasis: dict, n: int) -> tuple[np.ndarray, _CovarianceDict]:
    tp, _ = compute_interaction_matrix(quasis, n, "tunneling_plus"); tm, _ = compute_interaction_matrix(quasis, n, "tunneling_minus")
    sp, _ = compute_interaction_matrix(quasis, n, "superconducting_plus"); sm, _ = compute_interaction_matrix(quasis, n, "superconducting_minus")
    tunneling, superconducting = 0.5 * (tp + 1j * tm), 0.5 * (sp + 1j * sm)
    corr = np.block([[tunneling, superconducting], [-superconducting.conj(), np.eye(n) - tunneling.T]])
    if num_q := quasis.get((tuple(range(n)), "number")):
        for i in range(n):
            corr[i, i] = expval(num_q, "I"*(n-i-1)+"1"+"I"*i, n); corr[i+n, i+n] = 1 - corr[i, i]
    return corr, {}

def compute_correlation_matrix_exact(n_modes, tunneling, superconducting, chemical_potential, occupied_orbitals) -> tuple[np.ndarray, None]:
    trans_mat,_,_,_ = diagonalizing_bogoliubov_transform(n_modes, tunneling, superconducting, chemical_potential)
    W1, W2 = trans_mat[:, :n_modes], trans_mat[:, n_modes:]
    full_trans_mat = np.block([[W1, W2], [W2.conj(), W1.conj()]])
    occ = np.zeros(n_modes); occ[list(occupied_orbitals)] = 1.0
    corr_diag = np.diag(np.concatenate([occ, 1 - occ]))
    return full_trans_mat.T.conj() @ corr_diag @ full_trans_mat, None

def get_exact_energies(n_modes, tunneling, superconducting, chemical_potential_values, occupied_orbitals_list) -> dict:
    energies = defaultdict(list)
    for mu in chemical_potential_values:
        _, orb_e, const, _ = diagonalizing_bogoliubov_transform(n_modes, tunneling, superconducting, mu)
        for oo in occupied_orbitals_list:
            energies[oo].append(np.sum(orb_e[list(oo)]) + const)
    return {k: np.array(v) for k, v in energies.items()}

def _all_real_rz_gates(circuit: QuantumCircuit, rtol=1e-5, atol=1e-8) -> bool:
    for instruction in circuit.data:
        gate = instruction.operation
        if isinstance(gate, XXPlusYYGate):
            _, beta = gate.params
            if not np.isclose((beta + np.pi / 2) % np.pi, 0.0, rtol=rtol, atol=atol) and \
               not np.isclose((beta + np.pi / 2) % np.pi, np.pi, atol=atol):
                return False
    return True


def purify_idempotent_matrix(
        mat: np.ndarray, tol: float = 1e-8, max_iter: int = 100
) -> np.ndarray:
    """
    Implements McWeeny purification (rho_new = 3*rho^2 - 2*rho^3).
    Enforces the physical constraint that the correlation matrix should be
    idempotent (its eigenvalues are 0 or 1).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    clipped_eigenvalues = np.clip(np.real(eigenvalues), 0, 1)
    stable_mat = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T.conj()

    dim, _ = stable_mat.shape
    three = 3 * np.eye(dim, dtype=stable_mat.dtype)
    current_mat = stable_mat

    for _ in range(max_iter):
        mat_sq = current_mat @ current_mat
        new_mat = mat_sq @ (three - 2 * current_mat)

        if not np.all(np.isfinite(new_mat)):
            return current_mat

        error = np.linalg.norm(new_mat @ new_mat - new_mat)
        if error < tol:
            return new_mat
        current_mat = new_mat

    return current_mat


def generate_circuits(n_modes: int, tunneling: float, superconducting: float, chemical_potential: float,
                      occupied_orbitals: tuple[int, ...]) -> dict:
    circuits = {}

    transformation_matrix_base, _, _, _ = diagonalizing_bogoliubov_transform(
        n_modes, tunneling, superconducting, chemical_potential
    )

    for permutation, label in measurement_labels(n_modes):
        transformation_matrix = transformation_matrix_base.copy()
        perm = np.array(permutation)
        full_permutation = np.concatenate([perm, perm + n_modes])
        transformation_matrix = transformation_matrix[:, full_permutation]
        base_circuit = FermionicGaussianState(transformation_matrix, occupied_orbitals)

        if "_minus_" in label and _all_real_rz_gates(base_circuit):
            continue

        circuits[(permutation, label)] = measure_interaction_op(base_circuit, label)

    return circuits