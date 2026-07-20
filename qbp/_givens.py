"""Non-interacting (Givens) initial-state preparation for VQE warm starts.

The ground state of the one-body part of a fermionic Hamiltonian is a Slater
determinant. It is obtained classically in O(N^3) time by diagonalizing the
N x N one-body matrix, and prepared on the device by a network of
O(N * n_occ) Givens rotations between adjacent Jordan-Wigner modes. No step scales exponentially.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def one_body_matrix(fermionic_hamiltonian, n_orbitals: int) -> np.ndarray:
    """The N x N one-body (non-interacting) matrix of a FermionicOp.

    Collects ``+_i -_j`` terms into ``h[i, j]``; ``-_i +_j`` terms contribute
    ``-h[j, i]`` (their normal-ordering constant only shifts the energy and is
    irrelevant to the state). Higher-order (interaction) terms are ignored.
    """
    h = np.zeros((n_orbitals, n_orbitals), dtype=complex)
    for label, coeff in fermionic_hamiltonian.items():
        actions = label.split()
        if len(actions) != 2:
            continue
        (a_kind, a_idx), (b_kind, b_idx) = (a.split("_") for a in actions)
        i, j = int(a_idx), int(b_idx)
        if (a_kind, b_kind) == ("+", "-"):
            h[i, j] += coeff
        elif (a_kind, b_kind) == ("-", "+"):
            h[j, i] -= coeff
    return h


def _staircase_basis(occupied: np.ndarray) -> np.ndarray:
    """Rebase the occupied-orbital rows so row m is supported on modes
    ``0 .. N - M + m`` (a free left rotation; the state is unchanged)."""
    m_occ, n = occupied.shape
    ws: list[np.ndarray] = []
    for m in range(m_occ):
        tail = occupied[:, n - m_occ + m + 1:]
        if tail.shape[1]:
            _, s, vh = np.linalg.svd(tail.conj().T, full_matrices=True)
            rank = int(np.sum(s > 1e-12))
            candidates = vh[rank:].conj().T
        else:
            candidates = np.eye(m_occ, dtype=complex)
        for w_prev in ws:
            candidates = candidates - np.outer(w_prev, w_prev.conj() @ candidates)
        norms = np.linalg.norm(candidates, axis=0)
        best = int(np.argmax(norms))
        if norms[best] < 1e-12:
            raise ValueError("Failed to construct staircase basis for Slater determinant.")
        ws.append(candidates[:, best] / norms[best])
    w_mat = np.column_stack(ws)
    return w_mat.conj().T @ occupied


def _givens_eliminations(u: np.ndarray):
    """Right-multiplied adjacent-column Givens rotations bringing the
    staircase matrix ``u`` to the form ``[T | 0]``. Yields ``(j, g)`` pairs
    meaning columns ``(j - 1, j)`` were mixed by the 2x2 unitary ``g``."""
    m_occ, n = u.shape
    rotations = []
    for m in range(m_occ):
        for j in range(n - m_occ + m, m, -1):
            x, y = u[m, j - 1], u[m, j]
            rho = np.hypot(abs(x), abs(y))
            if abs(y) < 1e-14:
                continue
            g = np.array([
                [np.conj(x) / rho, -y / rho],
                [np.conj(y) / rho, x / rho],
            ])
            u[:, [j - 1, j]] = u[:, [j - 1, j]] @ g
            u[m, j] = 0.0
            rotations.append((j, g))
    return rotations


def _givens_gate(u2: np.ndarray) -> UnitaryGate:
    """Number-conserving 2-qubit gate implementing the single-particle
    unitary ``u2`` on two adjacent Jordan-Wigner modes."""
    mat = np.array([
        [1, 0, 0, 0],
        [0, u2[0, 0], u2[0, 1], 0],
        [0, u2[1, 0], u2[1, 1], 0],
        [0, 0, 0, np.linalg.det(u2)],
    ])
    return UnitaryGate(mat, label="givens")


def free_fermion_prep(h: np.ndarray, n_occ: int) -> QuantumCircuit:
    """Circuit preparing the ground state of the one-body matrix ``h`` at
    filling ``n_occ``, assuming Jordan-Wigner ordering (mode i -> qubit i)."""
    n = h.shape[0]
    qc = QuantumCircuit(n)
    for i in range(n_occ):
        qc.x(i)
    if n_occ == 0 or n_occ == n:
        return qc
    _, eigvecs = np.linalg.eigh(h)
    occupied = eigvecs[:, :n_occ].T
    stair = _staircase_basis(occupied)
    rotations = _givens_eliminations(stair)
    for j, g in reversed(rotations):
        qc.append(_givens_gate(np.conj(g)), [j - 1, j])
    return qc
