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

import functools
import math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
from qiskit_nature.second_q.operators import FermionicOp

def orbital_combinations(
    n_modes: int, threshold: Optional[int] = None
) -> Iterable[tuple[int, ...]]:
    if threshold is None:
        threshold = n_modes
    yield ()
    yield tuple(range(n_modes))
    for i in range(threshold):
        yield (i,)
        yield tuple(range(i)) + tuple(range(i + 1, n_modes))

def orbital_permutations(n_modes: int) -> Iterable[tuple[int, ...]]:
    permutation = list(range(n_modes))
    for _ in range(math.ceil(n_modes / 2)):
        yield tuple(permutation)
        for i in range(0, n_modes - 1, 2):
            a, b = permutation[i], permutation[i + 1]
            permutation[i], permutation[i + 1] = b, a
        for i in range(1, n_modes - 1, 2):
            a, b = permutation[i], permutation[i + 1]
            permutation[i], permutation[i + 1] = b, a

def measurement_labels(n_modes: int) -> Iterable[tuple[tuple[int, ...], str]]:
    yield tuple(range(n_modes)), "number"
    for permutation in orbital_permutations(n_modes):
        yield permutation, "tunneling_plus_even"
        yield permutation, "tunneling_plus_odd"
        yield permutation, "tunneling_minus_even"
        yield permutation, "tunneling_minus_odd"
        yield permutation, "superconducting_plus_even"
        yield permutation, "superconducting_plus_odd"
        yield permutation, "superconducting_minus_even"
        yield permutation, "superconducting_minus_odd"

@functools.lru_cache
def kitaev_hamiltonian(
    n_modes: int,
    tunneling: float,
    superconducting: Union[float, complex],
    chemical_potential: float,
) -> QuadraticHamiltonian:
    eye = np.eye(n_modes)
    upper_diag = np.diag(np.ones(n_modes - 1), k=1)
    lower_diag = np.diag(np.ones(n_modes - 1), k=-1)
    hermitian_part = -tunneling * (upper_diag + lower_diag) + chemical_potential * eye
    antisymmetric_part = superconducting * (upper_diag - lower_diag)
    constant = -0.5 * chemical_potential * n_modes
    return QuadraticHamiltonian(
        hermitian_part=hermitian_part,
        antisymmetric_part=antisymmetric_part,
        constant=constant,
    )

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
def diagonalizing_bogoliubov_transform(
    n_modes: int,
    tunneling: float,
    superconducting: Union[float, complex],
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    return kitaev_hamiltonian(
        n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential=chemical_potential,
    ).diagonalizing_bogoliubov_transform()

n_modes = 6
tunneling = -1.0
superconducting = 1.0
chemical_potential_values = [1.5] #list(np.linspace(0.0, 3.0, num=50))
occupied_orbitals_list = list(orbital_combinations(n_modes, threshold=2))

def data_exact(n_modes: int, tunneling: float, superconducting: float, chemical_potential_values: list[int], occupied_orbitals_list: list[tuple[int]]) -> dict[str, dict[tuple[int, ...], list[float]]]:
    start = chemical_potential_values[0]
    if start == 0:
        start = 1e-8

    edge_correlation = edge_correlation_op(n_modes)
    energy_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
    edge_correlation_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
    parity_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
    number_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]

    data = {}
    for chemical_potential in chemical_potential_values:
        (
            transformation_matrix,
            orbital_energies,
            constant,
        ) = diagonalizing_bogoliubov_transform(
            n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        W1 = transformation_matrix[:, : n_modes]
        W2 = transformation_matrix[:, n_modes :]
        full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
        hamiltonian_parity = np.sign(
            np.real(np.linalg.det(full_transformation_matrix))
        )
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
            exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
            exact_number = np.real(np.sum(np.diag(corr_exact)[: n_modes]))

            energy_exact[occupied_orbitals].append(exact_energy)
            edge_correlation_exact[occupied_orbitals].append(exact_edge_correlation)
            parity_exact[occupied_orbitals].append(exact_parity)
            number_exact[occupied_orbitals].append(exact_number)

    def zip_dict(d):
        return {k: (np.array(v), chemical_potential_values) for k, v in d.items()}

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
            particle = np.array(energy_exact[combs[2 * i + 2]])
            hole = np.array(energy_exact[combs[2 * i + 3]])
            bdg_energy[i] = particle - low
            bdg_energy[threshold + i] = hole - high
        data["bdg_energy_exact"] = (bdg_energy, chemical_potential_values)

    site_correlation_ops = [
        site_correlation_op(i) for i in range(1, 2 * n_modes)
    ]
    site_correlation_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
    for chemical_potential in chemical_potential_values:
        (
            transformation_matrix,
            orbital_energies,
            constant,
        ) = diagonalizing_bogoliubov_transform(
            n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        W1 = transformation_matrix[:, : n_modes]
        W2 = transformation_matrix[:, n_modes :]
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