from __future__ import annotations

from typing import Callable


class ModelCapabilityError(Exception):
    pass


def matrix_to_fermionic_op(H, tol: float = 1e-12):
    from qiskit_nature.second_q.operators import FermionicOp
    terms: dict[str, complex] = {}
    N = H.shape[0]
    for i in range(N):
        for j in range(N):
            c = complex(H[i, j])
            if abs(c) > tol:
                terms[f"+_{i} -_{j}"] = c
    return FermionicOp(terms, num_spin_orbitals=N)


class Model:
    def __init__(
        self,
        name: str,
        display_name: str,
        param_labels: dict[str, str],
        *,
        hamiltonian_matrix: Callable,
        interaction_hamiltonian: Callable | None = None,
        get_optimizer: Callable | None = None,
        mean_field_correction: Callable | None = None,
        bloch_hamiltonian: Callable | None = None,
        momentum_axes: tuple[str, ...] = (),
    ):
        if hamiltonian_matrix is None:
            raise ValueError(
                f"Model '{name}' requires hamiltonian_matrix."
            )
        self.name = name
        self.display_name = display_name
        self.param_labels = param_labels
        self.momentum_axes = tuple(momentum_axes)

        self._hamiltonian_matrix_fn = hamiltonian_matrix
        self._interaction_hamiltonian_fn = interaction_hamiltonian
        self._get_optimizer_fn = get_optimizer
        self._mean_field_correction_fn = mean_field_correction
        self._bloch_hamiltonian_fn = bloch_hamiltonian

    @property
    def _build_H_matrix(self):
        return self._hamiltonian_matrix_fn

    @property
    def fermionic_hamiltonian(self):
        def _build(n_sites, **params):
            H = self._hamiltonian_matrix_fn(n_sites, **params)
            op = matrix_to_fermionic_op(H)
            if self._interaction_hamiltonian_fn is not None:
                op = op + self._interaction_hamiltonian_fn(n_sites, **params)
            return op
        return _build

    @property
    def get_optimizer(self):
        if self._get_optimizer_fn is not None:
            return self._get_optimizer_fn
        from qiskit_algorithms.optimizers import SPSA
        return lambda max_iters: SPSA(maxiter=max_iters)

    @property
    def mean_field_correction(self):
        return self._mean_field_correction_fn

    @property
    def bloch_hamiltonian(self):
        if self._bloch_hamiltonian_fn is None:
            raise ModelCapabilityError(
                f"Model '{self.name}' does not implement bloch_hamiltonian; "
                f"momentum-space (band structure) runs are not supported."
            )
        return self._bloch_hamiltonian_fn

    @property
    def supports_band_structure(self) -> bool:
        return self._bloch_hamiltonian_fn is not None

    @property
    def NAME(self):
        return self.name

    @property
    def DISPLAY_NAME(self):
        return self.display_name

    @property
    def PARAM_LABELS(self):
        return self.param_labels

    def __repr__(self):
        caps = ["analytic", "simulation"]
        if self._interaction_hamiltonian_fn is not None:
            caps.append("interacting")
        if self._bloch_hamiltonian_fn is not None:
            caps.append("band-structure")
        return f"Model(name={self.name!r}, capabilities={caps})"
