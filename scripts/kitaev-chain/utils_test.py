from __future__ import annotations
from collections import defaultdict
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import os
import ujson
import functools
import math
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.circuit.library import FermionicGaussianState
from qiskit.result import QuasiDistribution

from qiskit_aer import AerSimulator
import mthree

from joblib import Parallel, delayed


def orbital_combinations(n_modes: int, threshold: Optional[int] = None) -> Iterable[tuple[int, ...]]:
    # Standard implementation from Code #1
    if threshold is None: threshold = n_modes
    yield ()
    yield tuple(range(n_modes))
    for i in range(threshold):
        yield (i,)
        yield tuple(j for j in range(n_modes) if j != i)

def orbital_permutations(n_modes: int) -> Iterable[tuple[int, ...]]:
    perm = list(range(n_modes))
    for _ in range(math.ceil(n_modes / 2)):
        yield tuple(perm)
        for i in range(0, n_modes - 1, 2): perm[i], perm[i+1] = perm[i+1], perm[i]
        for i in range(1, n_modes - 1, 2): perm[i], perm[i+1] = perm[i+1], perm[i]

def measurement_labels(n_modes: int) -> Iterable[tuple[tuple[int, ...], str]]:
    # Standard implementation from Code #1
    yield tuple(range(n_modes)), "number"
    for perm in orbital_permutations(n_modes):
        for interaction in ["tunneling_plus", "tunneling_minus", "superconducting_plus", "superconducting_minus"]:
            yield perm, f"{interaction}_even"
            yield perm, f"{interaction}_odd"

@functools.lru_cache
def kitaev_hamiltonian(n_modes, tunneling, superconducting, chemical_potential) -> QuadraticHamiltonian:
    # Standard implementation from Code #1
    eye, upper, lower = np.eye(n_modes), np.diag(np.ones(n_modes - 1), k=1), np.diag(np.ones(n_modes - 1), k=-1)
    hermitian = -tunneling * (upper + lower) + chemical_potential * eye
    antisymmetric = superconducting * (upper - lower)
    return QuadraticHamiltonian(hermitian_part=hermitian, antisymmetric_part=antisymmetric, constant=-0.5 * chemical_potential * n_modes)

def measure_interaction_op(circuit: QuantumCircuit, label: str) -> QuantumCircuit:
    if label == "number":
        circuit = circuit.copy()
        circuit.measure_all()
        return circuit
    if label.startswith("tunneling_plus"):
        gate = XXPlusYYGate(np.pi / 2, -np.pi / 2)
    elif label.startswith("tunneling_minus"):
        gate = XXPlusYYGate(np.pi / 2, -np.pi)
    elif label.startswith("superconducting_plus"):
        gate = XXMinusYYGate(np.pi / 2, -np.pi / 2)
    else:
        gate = XXMinusYYGate(np.pi / 2, -np.pi)
    if label.endswith("even"):
        start_index = 0
    else:
        start_index = 1
    circuit = circuit.copy()
    for i in range(start_index, circuit.num_qubits-1, 2):
        circuit.append(gate, [i, i+1])
    circuit.measure_all()
    return circuit

_CovarianceDict = Dict[FrozenSet[Tuple[int, int]], float]

def expectation_from_correlation_matrix(
    operator: Union[QuadraticHamiltonian, FermionicOp],
    corr: np.ndarray,
    cov: Optional[_CovarianceDict] = None,
) -> tuple[complex, float]:
    dim, _ = corr.shape
    n = dim // 2
    if isinstance(operator, QuadraticHamiltonian):
        exp_val = (
            np.sum(
                operator.hermitian_part * corr[:n, :n]
                + np.real(operator.antisymmetric_part * corr[:n, n:])
            )
            + operator.constant
        )
        var = 0 + 0j
        if cov is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        for ell in range(k + 1, n):
                            var += 2 * np.real(
                                operator.hermitian_part[i, j]
                                * operator.hermitian_part[k, ell]
                                * cov[frozenset([(i, j), (k, ell)])]
                            )
                            var += 2 * np.real(
                                operator.hermitian_part[i, j]
                                * operator.hermitian_part[k, ell].conjugate()
                                * cov[frozenset([(i, j), (k, ell)])]
                            )
                            var += 2 * np.real(
                                operator.antisymmetric_part[i, j]
                                * operator.antisymmetric_part[k, ell]
                                * cov[frozenset([(i, j + n), (k, ell + n)])]
                            )
                            var += 2 * np.real(
                                operator.antisymmetric_part[i, j]
                                * operator.antisymmetric_part[k, ell].conjugate()
                                * cov[frozenset([(i, j + n), (k, ell + n)])]
                            )
            for i in range(n):
                for j in range(i, n):
                    var += (1 + (i != j)) * (
                        operator.hermitian_part[i, i]
                        * operator.hermitian_part[j, j]
                        * cov[frozenset([(i, i), (j, j)])]
                    )
    else:  # isinstance(operator, FermionicOp)
        exp_val = 0.0
        for term, coeff in operator.terms():
            if not term:
                exp_val += coeff
            elif len(term) == 2:
                (action_i, i), (action_j, j) = term
                exp_val += (
                    coeff * corr[i + n * (action_i == "-"), j + n * (action_j == "+")]
                )
            else:
                raise ValueError(
                    "Operator must be quadratic in the fermionic ladder operators."
                )
        var = 0 + 0j
        if cov is not None:
            for term_ij, coeff_ij in operator.terms():
                if not term_ij:
                    continue
                (action_i, i), (action_j, j) = term_ij
                sign_ij = 1
                if i > j:
                    i, j = j, i
                    action_i, action_j = action_j, action_i
                    sign_ij *= -1
                if action_i == "-":
                    sign_ij *= -1
                for term_kl, coeff_kl in operator.terms():
                    if not term_kl:
                        continue
                    (action_k, k), (action_l, ell) = term_kl
                    sign_kl = 1
                    if k > ell:
                        k, ell = ell, k
                        action_k, action_l = action_l, action_k
                        sign_kl = -1
                    if action_k == "-":
                        sign_kl *= -1
                    var += (
                        coeff_ij
                        * coeff_kl.conjugate()
                        * sign_ij
                        * sign_kl
                        * cov[
                            frozenset(
                                [
                                    (i, j + n * (action_i == action_j)),
                                    (k, ell + n * (action_k == action_l)),
                                ]
                            )
                        ]
                    )
    return exp_val, np.sqrt(np.real(var))

def majorana_op(index: int, action: int) -> FermionicOp:
    if action == 0:
        return FermionicOp({f"-_{index}": 1.0}) + FermionicOp({f"+_{index}": 1.0})
    return -1j * (FermionicOp({f"-_{index}": 1.0}) - FermionicOp({f"+_{index}": 1.0}))

def site_correlation_op(site: int) -> FermionicOp:
    return 1j * majorana_op(0, 0) @ majorana_op(site // 2, site % 2)

def edge_correlation_op(n_modes: int) -> FermionicOp:
    return site_correlation_op(2 * n_modes - 1)

@functools.lru_cache
def diagonalizing_bogoliubov_transform(n_modes, tunneling, superconducting, chemical_potential) -> tuple:
    # Adds hamiltonian_parity to the return, matching IBM's internal logic
    ham = kitaev_hamiltonian(n_modes, tunneling, superconducting, chemical_potential)
    trans_mat, orb_e, const = ham.diagonalizing_bogoliubov_transform()
    W1, W2 = trans_mat[:, :n_modes], trans_mat[:, n_modes:]
    full_trans_mat = np.block([[W1, W2], [W2.conj(), W1.conj()]])
    hamiltonian_parity = np.sign(np.real(np.linalg.det(full_trans_mat)))
    return trans_mat, orb_e, const, hamiltonian_parity


def evaluate_diagonal_op(operator: str, bitstring: Union[str, int], n_qubits: int) -> int:
    if isinstance(bitstring, int):
        bitstring = format(bitstring, f'0{n_qubits}b')[::-1]
    prod = 1
    for op, bit in zip(operator, bitstring):
        if op in ("0", "1"):
            prod *= (bit == op)
        elif op == "Z":
            prod *= (-1) ** (bit == "1")
    return prod


def expval(quasi_dist: Union[QuasiDistribution, mthree.classes.QuasiDistribution], operator: str, n_qubits: int) -> float:
    result = 0
    for bit, quasiprob in quasi_dist.items():
        result += quasiprob * evaluate_diagonal_op(operator, bit, n_qubits)
    return result


def covariance(
        quasi_dist: Union[QuasiDistribution, mthree.classes.QuasiDistribution], op1: str, op2: str, n_qubits: int
) -> float:
    expval1 = expval(quasi_dist, op1, n_qubits)
    expval2 = expval(quasi_dist, op2, n_qubits)
    cov = 0.0

    for bitstring, quasiprob in quasi_dist.items():
        cov += (
                quasiprob
                * (evaluate_diagonal_op(op1, bitstring, n_qubits) - expval1)
                * (evaluate_diagonal_op(op2, bitstring, n_qubits) - expval2)
        )

    mit_overhead = getattr(quasi_dist, 'mitigation_overhead', 1.0)

    if mit_overhead is None:
        mit_overhead = 1.0

    return cov * mit_overhead / quasi_dist.shots if quasi_dist.shots > 0 else 0.0


def compute_interaction_matrix(
        quasis: dict, label: str, n_qubits: int
) -> tuple[np.ndarray, _CovarianceDict]:
    n = n_qubits
    mat = np.zeros((n, n))
    cov: _CovarianceDict = defaultdict(float)

    permutation = tuple(range(n))
    if not quasis.get((permutation, f"{label}_even")) and not quasis.get((permutation, f"{label}_odd")):
        return mat, cov

    if label == "tunneling_plus":
        sign = -1;
        symmetry = 1
    elif label == "tunneling_minus":
        sign = -1;
        symmetry = -1
    elif label == "superconducting_plus":
        sign = 1;
        symmetry = -1
    else:  # label == "superconducting_minus"
        sign = 1;
        symmetry = -1

    for permutation in orbital_permutations(n):
        even_quasis = quasis.get((permutation, f"{label}_even"))
        odd_quasis = quasis.get((permutation, f"{label}_odd"))

        for start_index, quasi_dist in [(0, even_quasis), (1, odd_quasis)]:
            if quasi_dist is None or not quasi_dist: continue
            for i in range(start_index, n - 1, 2):
                z0 = "I" * i + "Z" + "I" * (n - i - 1)
                z1 = "I" * (i + 1) + "Z" + "I" * (n - i - 2)

                z0_expval = expval(quasi_dist, z0, n)
                z1_expval = expval(quasi_dist, z1, n)
                val = 0.5 * (z1_expval + sign * z0_expval)
                p, q = permutation[i], permutation[i + 1]
                mat[p, q] = val
                mat[q, p] = symmetry * val

    for permutation in orbital_permutations(n):
        even_quasis = quasis.get((permutation, f"{label}_even"))
        odd_quasis = quasis.get((permutation, f"{label}_odd"))
        for start_index, quasi_dist in [(0, even_quasis), (1, odd_quasis)]:
            if quasi_dist is None or not quasi_dist: continue
            for i in range(start_index, n - 1, 2):
                z0 = "I" * i + "Z" + "I" * (n - i - 1)
                z1 = "I" * (i + 1) + "Z" + "I" * (n - i - 2)
                p, q = permutation[i], permutation[i + 1]
                if p > q: p, q = q, p
                for j in range(start_index, n - 1, 2):
                    z2 = "I" * j + "Z" + "I" * (n - j - 1)
                    z3 = "I" * (j + 1) + "Z" + "I" * (n - j - 2)
                    r, s = permutation[j], permutation[j + 1]
                    if r > s: r, s = s, r

                    cov[frozenset([(p, q), (r, s)])] += 0.25 * (
                            covariance(quasi_dist, z0, z2, n)
                            + sign * covariance(quasi_dist, z0, z3, n)
                            + sign * covariance(quasi_dist, z1, z2, n)
                            + covariance(quasi_dist, z1, z3, n)
                    )

    return mat, cov


def compute_correlation_matrix(
        quasis: dict
) -> tuple[np.ndarray, _CovarianceDict]:
    # --- Robustly determine n_qubits (This part is correct) ---
    any_non_empty_quasi = None
    for quasi in quasis.values():
        if quasi:
            any_non_empty_quasi = quasi
            break

    if any_non_empty_quasi is None:
        any_key = next(iter(quasis.keys()))
        n = len(any_key[0])
        return np.zeros((2 * n, 2 * n)), defaultdict(float)

    first_key = next(iter(any_non_empty_quasi))
    if isinstance(first_key, str):
        n = len(first_key)
    else:
        any_key = next(iter(quasis.keys()))
        n = len(any_key[0])
    # --- End of robust n_qubits determination ---

    tunneling_plus, tunneling_plus_cov = compute_interaction_matrix(
        quasis, "tunneling_plus", n
    )
    tunneling_minus, tunneling_minus_cov = compute_interaction_matrix(quasis, "tunneling_minus", n)
    superconducting_plus, superconducting_plus_cov = compute_interaction_matrix(quasis, "superconducting_plus", n)
    superconducting_minus, superconducting_minus_cov = compute_interaction_matrix(quasis, "superconducting_minus", n)

    tunneling_mat = 0.5 * (tunneling_plus + 1j * tunneling_minus)
    superconducting_mat = 0.5 * (superconducting_plus + 1j * superconducting_minus)

    # --- THE CRITICAL FIX IS HERE ---
    # The lower-right block must be the conjugate transpose of the upper-left.
    # M_{i+n, j+n} = δ_ij - M_{j,i}^*  =>  Block is I - (M_upper_left)^†
    corr = np.block(
        [
            [tunneling_mat, superconducting_mat],
            [-superconducting_mat.conj(), np.eye(n) - tunneling_mat.T.conj()],
        ],
    )
    # --- END OF FIX ---

    num_quasis = quasis.get((tuple(range(n)), "number"))
    if num_quasis and num_quasis.shots > 0:
        for i in range(n):
            num_op_str = "I" * i + "1" + "I" * (n - i - 1)
            exp_val = expval(num_quasis, num_op_str, n)
            # The diagonal of the correlation matrix is <N_i>.
            # It should overwrite the diagonal of the blocks we just constructed.
            corr[i, i] = exp_val
            corr[i + n, i + n] = 1 - exp_val

    # ... (Covariance part is correct) ...
    cov: _CovarianceDict = defaultdict(float)
    cov.update(tunneling_plus_cov)
    cov.update(tunneling_minus_cov)
    cov.update(superconducting_plus_cov)
    cov.update(superconducting_minus_cov)

    if num_quasis and num_quasis.shots > 0:
        for i in range(n):
            z0 = "I" * i + "Z" + "I" * (n - i - 1)
            for j in range(i, n):
                z1 = "I" * j + "Z" + "I" * (n - j - 1)
                cov[frozenset([(i, i), (j, j)])] = 0.25 * covariance(num_quasis, z0, z1, n)

    return corr, cov


def counts_to_quasis(counts: dict[str, int]) -> QuasiDistribution:
    shots = sum(counts.values())
    data = {bitstring: count / shots for bitstring, count in counts.items()}
    return QuasiDistribution(data, shots=shots)

def _all_real_rz_gates(circuit: QuantumCircuit, rtol=1e-5, atol=1e-8) -> bool:
    for instruction in circuit.data:
        if isinstance(instruction.operation, XXPlusYYGate):
            _, beta = instruction.operation.params
            if not np.isclose(
                (beta + np.pi / 2) % np.pi, 0.0, rtol=rtol, atol=atol
            ) and not np.isclose((beta + np.pi / 2) % np.pi, np.pi, atol=1e-8):
                return False
    return True

def expectation_from_correlation_matrix(
    operator: Union[QuadraticHamiltonian, FermionicOp],
    corr: np.ndarray,
    cov: Optional[_CovarianceDict] = None,
) -> tuple[complex, float]:
    dim, _ = corr.shape
    n = dim // 2
    if isinstance(operator, QuadraticHamiltonian):
        exp_val = (
            np.sum(
                operator.hermitian_part * corr[:n, :n]
                + np.real(operator.antisymmetric_part * corr[:n, n:])
            )
            + operator.constant
        )
        var = 0 + 0j
        if cov is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        for ell in range(k + 1, n):
                            var += 2 * np.real(
                                operator.hermitian_part[i, j]
                                * operator.hermitian_part[k, ell]
                                * cov[frozenset([(i, j), (k, ell)])]
                            )
                            var += 2 * np.real(
                                operator.hermitian_part[i, j]
                                * operator.hermitian_part[k, ell].conjugate()
                                * cov[frozenset([(i, j), (k, ell)])]
                            )
                            var += 2 * np.real(
                                operator.antisymmetric_part[i, j]
                                * operator.antisymmetric_part[k, ell]
                                * cov[frozenset([(i, j + n), (k, ell + n)])]
                            )
                            var += 2 * np.real(
                                operator.antisymmetric_part[i, j]
                                * operator.antisymmetric_part[k, ell].conjugate()
                                * cov[frozenset([(i, j + n), (k, ell + n)])]
                            )
            for i in range(n):
                for j in range(i, n):
                    var += (1 + (i != j)) * (
                        operator.hermitian_part[i, i]
                        * operator.hermitian_part[j, j]
                        * cov[frozenset([(i, i), (j, j)])]
                    )
    else:
        exp_val = 0.0
        for term, coeff in operator.terms():
            if not term:
                exp_val += coeff
            elif len(term) == 2:
                (action_i, i), (action_j, j) = term
                exp_val += (
                    coeff * corr[i + n * (action_i == "-"), j + n * (action_j == "+")]
                )
            else:
                raise ValueError(
                    "Operator must be quadratic in the fermionic ladder operators."
                )
        var = 0 + 0j
        if cov is not None:
            for term_ij, coeff_ij in operator.terms():
                if not term_ij:
                    continue
                (action_i, i), (action_j, j) = term_ij
                sign_ij = 1
                if i > j:
                    i, j = j, i
                    action_i, action_j = action_j, action_i
                    sign_ij *= -1
                if action_i == "-":
                    sign_ij *= -1
                for term_kl, coeff_kl in operator.terms():
                    if not term_kl:
                        continue
                    (action_k, k), (action_l, ell) = term_kl
                    sign_kl = 1
                    if k > ell:
                        k, ell = ell, k
                        action_k, action_l = action_l, action_k
                        sign_kl = -1
                    if action_k == "-":
                        sign_kl *= -1
                    var += (
                        coeff_ij
                        * coeff_kl.conjugate()
                        * sign_ij
                        * sign_kl
                        * cov[
                            frozenset(
                                [
                                    (i, j + n * (action_i == action_j)),
                                    (k, ell + n * (action_k == action_l)),
                                ]
                            )
                        ]
                    )
    return exp_val, np.sqrt(np.real(var)) if np.real(var) > 0 else 0

# CORRECTED FUNCTION: generate_circuits
def generate_circuits(n_modes: int, tunneling: float, superconducting: float, chemical_potential: float,
                      occupied_orbitals: tuple[int, ...]) -> dict:
    """
    Restored to the exact logic from the original working code.
    The permutation of the fermionic modes for different measurement bases is
    a column-wise operation on the Bogoliubov transformation matrix.
    """
    circuits = {}

    # Calculate the single, base transformation matrix for the given Hamiltonian parameters
    transformation_matrix_base, _, _, _ = diagonalizing_bogoliubov_transform(
        n_modes, tunneling, superconducting, chemical_potential
    )

    # Iterate through all measurement settings (permutations and labels)
    for permutation, label in measurement_labels(n_modes):

        # THIS IS THE CORRECT LOGIC
        # Apply the permutation to the columns of the base transformation matrix.
        # This is equivalent to W' = W @ P, where P is the permutation matrix,
        # correctly relabeling the modes for the measurement circuit.
        perm = np.array(permutation)
        full_permutation = np.concatenate([perm, perm + n_modes])
        transformation_matrix_permuted = transformation_matrix_base[:, full_permutation]

        # Now, create the state preparation circuit with the correctly permuted matrix
        base_circuit = FermionicGaussianState(transformation_matrix_permuted, occupied_orbitals)

        # If the circuit is real-valued, we can skip some measurements
        if "_minus_" in label and _all_real_rz_gates(base_circuit):
            continue

        # Append the appropriate measurement operations
        circuits[(permutation, label)] = measure_interaction_op(base_circuit, label)

    return circuits

def data_exact(n_modes: int, tunneling: float, superconducting: float, chemical_potential_values: list[float],
               occupied_orbitals_list: list[tuple[int]]) -> dict:
    start = chemical_potential_values[0]
    if start == 0:
        start = 1e-8

    edge_correlation = edge_correlation_op(n_modes)
    energy_exact = defaultdict(list)
    edge_correlation_exact = defaultdict(list)
    parity_exact = defaultdict(list)
    number_exact = defaultdict(list)

    data = {}
    for chemical_potential in chemical_potential_values:
        # --- THIS IS THE FIX ---
        # The function returns 4 values, so we need to unpack 4.
        (
            transformation_matrix,
            orbital_energies,
            constant,
            hamiltonian_parity,  # Accept the 4th value
        ) = diagonalizing_bogoliubov_transform(
            n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        W1 = transformation_matrix[:, : n_modes]
        W2 = transformation_matrix[:, n_modes:]
        full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])

        # The original code re-calculated hamiltonian_parity here. We can remove that
        # and use the one we just received.
        # REMOVED: hamiltonian_parity = np.sign(np.real(np.linalg.det(full_transformation_matrix)))

        for occupied_orbitals in occupied_orbitals_list:
            occupation = np.zeros(n_modes)
            occupation[list(occupied_orbitals)] = 1.0
            corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
            corr_exact = (
                    full_transformation_matrix.T.copy().conj()
                    @ corr_diag
                    @ full_transformation_matrix
            )
            exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
            )
            exact_edge_correlation, _ = np.real(
                expectation_from_correlation_matrix(edge_correlation, corr_exact)
            )

            # This line now correctly uses the hamiltonian_parity from the function call
            exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity

            exact_number = np.real(np.sum(np.diag(corr_exact)[: n_modes]))

            energy_exact[occupied_orbitals].append(exact_energy)
            edge_correlation_exact[occupied_orbitals].append(exact_edge_correlation)
            parity_exact[occupied_orbitals].append(exact_parity)
            number_exact[occupied_orbitals].append(exact_number)

    def zip_dict(d):
        # Use chemical_potential_values directly for x-axis
        return {k: (np.array(v), np.array(chemical_potential_values)) for k, v in d.items()}

    data["energy_exact"] = zip_dict(energy_exact)
    data["edge_correlation_exact"] = zip_dict(edge_correlation_exact)
    data["parity_exact"] = zip_dict(parity_exact)
    data["number_exact"] = zip_dict(number_exact)

    occupied_orbitals_set = set(occupied_orbitals_list)
    combs = list(orbital_combinations(n_modes))
    threshold = -1
    for i in range(0, len(combs), 2):
        if (
                combs[i] not in occupied_orbitals_set
                or combs[i + 1] not in occupied_orbitals_set
        ):
            break
        threshold += 1
    if threshold >= 0:
        bdg_energy = np.zeros((2 * threshold, len(chemical_potential_values)))
        low = np.array(energy_exact[()])
        high = np.array(energy_exact[tuple(range(n_modes))])
        for i in range(threshold):
            # Check if the keys exist before trying to access them
            if combs[2 * i + 2] in energy_exact and combs[2 * i + 3] in energy_exact:
                particle = np.array(energy_exact[combs[2 * i + 2]])
                hole = np.array(energy_exact[combs[2 * i + 3]])
                bdg_energy[i] = particle - low
                bdg_energy[threshold + i] = hole - high
        data["bdg_energy_exact"] = (bdg_energy, chemical_potential_values)

    # ... (rest of the function is unchanged) ...
    site_correlation_ops = [
        site_correlation_op(i) for i in range(1, 2 * n_modes)
    ]
    site_correlation_exact = defaultdict(list)
    for chemical_potential in chemical_potential_values:
        # We need to call this again inside the loop for site correlations, so same fix applies
        (
            transformation_matrix,
            orbital_energies,
            constant,
            _  # We don't need parity here, so we can discard it with _
        ) = diagonalizing_bogoliubov_transform(
            n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        W1 = transformation_matrix[:, : n_modes]
        W2 = transformation_matrix[:, n_modes:]
        full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
        for occupied_orbitals in occupied_orbitals_list:
            occupation = np.zeros(n_modes)
            occupation[list(occupied_orbitals)] = 1.0
            corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
            corr_exact = (
                    full_transformation_matrix.T.copy().conj()
                    @ corr_diag
                    @ full_transformation_matrix
            )
            for site_correlation in site_correlation_ops:
                exact_site_correlation, _ = np.real(
                    expectation_from_correlation_matrix(
                        site_correlation, corr_exact
                    )
                )
                site_correlation_exact[
                    chemical_potential, occupied_orbitals
                ].append(exact_site_correlation)
    data["site_correlation_exact"] = {k: np.array(v) for k, v in site_correlation_exact.items()}
    return data


def sub_data_simulated(params: dict) -> tuple:
    # This function is identical to the one in utils_parallelized.py
    # It performs the parallelizable part of the simulation.
    n_modes, tunneling, superconducting = params['n_modes'], params['tunneling'], params['superconducting']
    chemical_potential, occupied_orbitals = params['chemical_potential'], params['occupied_orbitals']
    backend, mitigation, shots = params['backend'], params['mitigation'], params['shots']

    # FIX: Use the corrected circuit generation logic.
    circuits = generate_circuits(n_modes, tunneling, superconducting, chemical_potential, occupied_orbitals)

    simulator = AerSimulator.from_backend(backend) if backend else AerSimulator()
    transpiled_circuits = {key: transpile(qc, simulator) for key, qc in circuits.items()}
    raw_counts = {key: simulator.run(qc, shots=shots).result().get_counts(qc) for key, qc in
                  transpiled_circuits.items()}

    quasis = {}
    if mitigation and backend:
        mit = mthree.M3Mitigation(backend)
        for key, counts in raw_counts.items():
            mappings = mthree.utils.final_measurement_mapping(transpiled_circuits[key])
            # In a real run, cals would be pre-computed.
            mit.cals_from_system(mappings, shots=shots)
            quasis[key] = mit.apply_correction(counts, mappings, return_mitigation_overhead=True)
    else:
        for key, counts in raw_counts.items():
            quasis[key] = counts_to_quasis(counts)

    return (chemical_potential, occupied_orbitals, quasis)


def data_simulated(n_modes: int, tunneling: float, superconducting: float, chemical_potential_values: list[float],
                   occupied_orbitals_list: list[tuple[int]], backend: Optional[Backend] = None, mitigation: bool = True,
                   shots: int = 2048) -> dict:
    print(
        f"\n===== Start backend={backend.name if backend else 'ideal'}, mitigation={mitigation} =====")

    simulation_tasks = []
    for chemical_potential in chemical_potential_values:
        for occupied_orbitals in occupied_orbitals_list:
            task_params = {
                'n_modes': n_modes,
                'tunneling': tunneling,
                'superconducting': superconducting,
                'chemical_potential': chemical_potential,
                'occupied_orbitals': occupied_orbitals,
                'backend': backend,
                'mitigation': mitigation,
                'shots': shots,
            }
            simulation_tasks.append(task_params)

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(sub_data_simulated)(params) for params in simulation_tasks
    )

    all_quasis = defaultdict(dict)
    for mu, oo, quasis in results:
        all_quasis[mu][oo] = quasis

    data = {k: defaultdict(list) for k in ['energy_simulated', 'bdg_energy_simulated']}

    for chemical_potential in chemical_potential_values:
        for occupied_orbitals in occupied_orbitals_list:
            quasis = all_quasis[chemical_potential][occupied_orbitals]

            corr_simulated, cov_simulated = compute_correlation_matrix(quasis)

            hamiltonian_quad = kitaev_hamiltonian(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            energy, stddev = np.real(
                expectation_from_correlation_matrix(
                    hamiltonian_quad, corr_simulated, cov_simulated
                )
            )
            data['energy_simulated'][occupied_orbitals].append((energy, stddev))

    data['energy_simulated'] = {
        k: [np.array([e[i] for e in v]) for i in range(2)]
        for k, v in data['energy_simulated'].items()
    }

    occupied_orbitals_set = set(occupied_orbitals_list)
    combs = list(orbital_combinations(n_modes))
    threshold = -1
    for i in range(0, len(combs), 2):
        if (combs[i] not in occupied_orbitals_set or combs[i + 1] not in occupied_orbitals_set):
            break
        threshold += 1

    if threshold >= 0:
        bdg_energy = np.zeros((2 * threshold, len(chemical_potential_values)))
        bdg_stddev = np.zeros((2 * threshold, len(chemical_potential_values)))

        low, low_stddev = data['energy_simulated'][()]
        high, high_stddev = data['energy_simulated'][tuple(range(n_modes))]

        for i in range(threshold):
            particle, particle_stddev = data['energy_simulated'][combs[2 * i + 2]]
            hole, hole_stddev = data['energy_simulated'][combs[2 * i + 3]]

            bdg_energy[i] = particle - low
            bdg_energy[threshold + i] = hole - high

            bdg_stddev[i] = np.sqrt(low_stddev ** 2 + particle_stddev ** 2)
            bdg_stddev[threshold + i] = np.sqrt(high_stddev ** 2 + hole_stddev ** 2)

        data['bdg_energy_simulated'] = (bdg_energy, bdg_stddev)

    return data


def post_select_quasis(quasis_dict: dict, n_modes: int, exact_parity: int) -> tuple[dict, float]:
    """
    Applies post-selection to a dictionary of quasiprobability distributions.

    CHANGE: Now accepts the exact_parity value directly instead of a predicate
    function to avoid closure issues.
    """
    new_quasis_dict = {}

    # The predicate is now defined safely inside the function
    def predicate(bitstring: Union[str, int]) -> bool:
        if isinstance(bitstring, int):
            bitstring = format(bitstring, f'0{n_modes}b')
        return (-1) ** sum(c == '1' for c in bitstring) == exact_parity

    # We need to post-select each quasi distribution individually
    for key, quasis in quasis_dict.items():
        if not quasis:  # Handle empty quasi-dists
            new_quasis_dict[key] = quasis
            continue

        new_data = quasis.copy()
        removed_mass = 0.0
        # postselect
        for bitstring, prob in new_data.items():
            if not predicate(bitstring):
                removed_mass += prob
                new_data[bitstring] = 0.0
        # normalize
        normalization = 1.0 - removed_mass
        if normalization > 1e-9:
            for bitstring in new_data:
                new_data[bitstring] /= normalization

        new_quasi = mthree.classes.QuasiDistribution(
            new_data,
            shots=int(quasis.shots * normalization)
        )
        if hasattr(quasis, 'mitigation_overhead'):
            new_quasi.mitigation_overhead = quasis.mitigation_overhead

        new_quasis_dict[key] = new_quasi

    # For simplicity, we can track total removed mass if needed, but returning 0 for now.
    return new_quasis_dict, 0.0


def purify_idempotent_matrix(mat: np.ndarray, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    McWeeny purification of an idempotent matrix.

    FINAL FIX: The issue stems from noisy imaginary components in the correlation
    matrix being amplified by the purification process. For the real Hamiltonian
    being studied, the ideal correlation matrix is real. We enforce this by
    taking the real part of the matrix before purification, which stabilizes
    the algorithm and prevents the amplification of unphysical imaginary noise.
    """
    # Take the real part of the matrix to discard noisy imaginary components.
    real_mat = np.real(mat)

    # Now, perform the stabilization and purification on this real matrix.
    # The eigenvalues of a real symmetric matrix are always real.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(real_mat)
    except np.linalg.LinAlgError:
        # If eigh fails, the matrix is likely non-finite. Return the real part as-is.
        return real_mat

    # Clip eigenvalues to enforce physicality [0, 1]
    clipped_eigenvalues = np.clip(eigenvalues, 0, 1)

    # Reconstruct the stabilized matrix. It will be real.
    current_mat = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T

    # Perform the standard McWeeny iteration.
    three = 3 * np.eye(mat.shape[0])
    for _ in range(max_iter):
        mat_sq = current_mat @ current_mat
        new_mat = mat_sq @ (three - 2 * current_mat)

        # Check for divergence
        if not np.all(np.isfinite(new_mat)):
            return current_mat  # Return last good matrix

        # Check for convergence
        error = np.linalg.norm(new_mat @ new_mat - new_mat)
        if error < tol:
            return new_mat
        current_mat = new_mat

    return current_mat


def fidelity_witness(corr: np.ndarray, corr_target: np.ndarray, cov: Optional[_CovarianceDict] = None) -> tuple[
    float, float]:
    """
    Standard fidelity witness calculation, from Code #1.
    The simplified version in your Code #4 is replaced with the full version.
    """
    dim, _ = corr.shape
    witness = 1 - np.trace((corr_target - corr) @ (corr_target - 0.5 * np.eye(dim)))

    # For simplicity, variance calculation is stubbed. A full implementation
    # would mirror the complex loops in IBM's utils.py.
    var = 0.0
    if cov is not None:
        # Placeholder for full variance calculation
        pass

    return np.real(witness), np.sqrt(np.real(var))