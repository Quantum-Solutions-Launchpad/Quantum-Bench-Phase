from __future__ import annotations

from typing import Callable


class ModelCapabilityError(Exception):
    pass


class Model:
    def __init__(
        self,
        name: str,
        display_name: str,
        param_labels: dict[str, str],
        *,
        default_params: dict[str, float] | None = None,
        hamiltonian_matrix: Callable | None = None,
        fermionic_hamiltonian: Callable | None = None,
        get_optimizer: Callable | None = None,
        mean_field_correction: Callable | None = None,
        sweep_defaults: dict | None = None,
    ):
        self.name = name
        self.display_name = display_name
        self.default_params = default_params or {}
        self.param_labels = param_labels
        self.sweep_defaults = sweep_defaults or {}

        self._hamiltonian_matrix_fn = hamiltonian_matrix
        self._fermionic_hamiltonian_fn = fermionic_hamiltonian
        self._get_optimizer_fn = get_optimizer
        self._mean_field_correction_fn = mean_field_correction

    @property
    def _build_H_matrix(self):
        if self._hamiltonian_matrix_fn is None:
            raise ModelCapabilityError(
                f"Model '{self.name}' does not provide hamiltonian_matrix; "
                "this is required for run_analytic(), run_simulated_ideal(), and run_simulated_noisy(). "
                "Provide hamiltonian_matrix= when constructing Model."
            )
        return self._hamiltonian_matrix_fn

    @property
    def fermionic_hamiltonian(self):
        if self._fermionic_hamiltonian_fn is None:
            raise ModelCapabilityError(
                f"Model '{self.name}' does not provide fermionic_hamiltonian; "
                "this is required for run_simulated_ideal() and run_simulated_noisy(). "
                "Provide fermionic_hamiltonian= when constructing Model."
            )
        return self._fermionic_hamiltonian_fn

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
        caps = []
        if self._hamiltonian_matrix_fn:
            caps.append("analytic")
        if self._fermionic_hamiltonian_fn:
            caps.append("simulation")
        return f"Model(name={self.name!r}, capabilities={caps})"
