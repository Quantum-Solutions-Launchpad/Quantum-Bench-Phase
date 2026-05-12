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
        default_params: dict[str, float] | None = None,
        interaction_hamiltonian: Callable | None = None,
        get_optimizer: Callable | None = None,
        mean_field_correction: Callable | None = None,
        sweep_defaults: dict | None = None,
    ):
        if hamiltonian_matrix is None:
            raise ValueError(
                f"Model '{name}' requires hamiltonian_matrix."
            )
        self.name = name
        self.display_name = display_name
        self.default_params = default_params or {}
        self.param_labels = param_labels
        self.sweep_defaults = sweep_defaults or {}

        self._hamiltonian_matrix_fn = hamiltonian_matrix
        self._interaction_hamiltonian_fn = interaction_hamiltonian
        self._get_optimizer_fn = get_optimizer
        self._mean_field_correction_fn = mean_field_correction

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
    def NAME(self):
        return self.name

    @property
    def DISPLAY_NAME(self):
        return self.display_name

    @property
    def DEFAULT_PARAMS(self):
        return self.default_params

    @property
    def PARAM_LABELS(self):
        return self.param_labels

    @property
    def SWEEP_DEFAULTS(self):
        return self.sweep_defaults

    def __repr__(self):
        caps = ["analytic", "simulation"]
        if self._interaction_hamiltonian_fn is not None:
            caps.append("interacting")
        return f"Model(name={self.name!r}, capabilities={caps})"
