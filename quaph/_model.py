from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ModelCapabilityError(Exception):
    pass


@dataclass
class Observable:
    name: str
    display_name: str
    analytic: Callable[..., float]
    analytic_bloch: Callable[..., Any] | None = None
    operator: Any = None


def _default_energy_analytic(model, lattice, H, eigvals, eigvecs, n_occ, params):
    import numpy as np
    e = float(np.sum(eigvals[:n_occ]))
    if model._mean_field_correction_fn is not None:
        e += float(model._mean_field_correction_fn(lattice, n_occ, **params))
    return e


def _default_energy_analytic_bloch(model, k_tuple, H, eigvals, eigvecs, params):
    import numpy as np
    return np.sort(eigvals).tolist()


def default_energy_observable() -> "Observable":
    return Observable(
        name="E",
        display_name="E",
        analytic=_default_energy_analytic,
        analytic_bloch=_default_energy_analytic_bloch,
    )


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
        spin: int,
        n_dims: int,
        lattice_shape: tuple[str, ...],
        sites_per_cell: int,
        hamiltonian_matrix: Callable,
        interaction_hamiltonian: Callable | None = None,
        get_optimizer: Callable | None = None,
        get_mapper: Callable | None = None,
        get_vqe_ansatz: Callable | None = None,
        mean_field_correction: Callable | None = None,
        bloch_hamiltonian: Callable | None = None,
        observables: dict[str, "Observable"] | None = None,
    ):
        if hamiltonian_matrix is None:
            raise ValueError(
                f"Model '{name}' requires hamiltonian_matrix."
            )
        if spin not in (1, 2):
            raise ValueError(
                f"Model '{name}' has invalid spin={spin}; must be 1 (spinless) or 2 (with spin)."
            )
        momentum_axes_by_dims = {1: ("k",), 2: ("kx", "ky"), 3: ("kx", "ky", "kz")}
        if n_dims not in momentum_axes_by_dims:
            raise ValueError(
                f"Model '{name}' has invalid n_dims={n_dims}; must be 1, 2, or 3."
            )
        lattice_shape = tuple(lattice_shape)
        if len(lattice_shape) != n_dims:
            raise ValueError(
                f"Model '{name}' has lattice_shape={lattice_shape} (len {len(lattice_shape)}) "
                f"but n_dims={n_dims}; they must match."
            )
        if not isinstance(sites_per_cell, int) or sites_per_cell < 1:
            raise ValueError(
                f"Model '{name}' has invalid sites_per_cell={sites_per_cell}; must be a positive int."
            )
        self.name = name
        self.display_name = display_name
        self.spin = spin
        self.n_dims = n_dims
        self.lattice_shape = lattice_shape
        self.sites_per_cell = sites_per_cell
        self.momentum_axes = momentum_axes_by_dims[n_dims]

        momentum_labels = {"k": "k", "kx": "k_x", "ky": "k_y", "kz": "k_z"}
        merged_labels = {a: momentum_labels[a] for a in self.momentum_axes}
        merged_labels.update(param_labels)
        self.param_labels = merged_labels

        self._hamiltonian_matrix_fn = hamiltonian_matrix
        self._interaction_hamiltonian_fn = interaction_hamiltonian
        self._get_optimizer_fn = get_optimizer
        self._get_mapper_fn = get_mapper
        self._get_vqe_ansatz_fn = get_vqe_ansatz
        self._mean_field_correction_fn = mean_field_correction
        self._bloch_hamiltonian_fn = bloch_hamiltonian

        merged_observables: dict[str, Observable] = {"E": default_energy_observable()}
        if observables:
            for obs_name, obs in observables.items():
                if obs.name != obs_name:
                    obs = Observable(
                        name=obs_name,
                        display_name=obs.display_name,
                        analytic=obs.analytic,
                        analytic_bloch=obs.analytic_bloch,
                        operator=obs.operator,
                    )
                merged_observables[obs_name] = obs
        self._observables = merged_observables

    @property
    def _build_H_matrix(self):
        return self._hamiltonian_matrix_fn

    @property
    def fermionic_hamiltonian(self):
        def _build(lattice, **params):
            H = self._hamiltonian_matrix_fn(lattice, **params)
            op = matrix_to_fermionic_op(H)
            if self._interaction_hamiltonian_fn is not None:
                op = op + self._interaction_hamiltonian_fn(lattice, **params)
            return op
        return _build

    @property
    def get_optimizer(self):
        if self._get_optimizer_fn is not None:
            return self._get_optimizer_fn
        from qiskit_algorithms.optimizers import SPSA
        return lambda max_iters: SPSA(maxiter=max_iters)

    @property
    def get_mapper(self):
        if self._get_mapper_fn is not None:
            return self._get_mapper_fn
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        return lambda n_sites, spin, n_occ: JordanWignerMapper()

    @property
    def get_vqe_ansatz(self):
        if self._get_vqe_ansatz_fn is not None:
            return self._get_vqe_ansatz_fn
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import excitation_preserving

        def default(n_qubits, n_layers, n_occ, spin):
            qc = QuantumCircuit(n_qubits)
            for i in range(n_occ):
                qc.x(i)
            qc.compose(
                excitation_preserving(n_qubits, "fsim", "linear", reps=n_layers),
                inplace=True,
            )
            return qc
        return default

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
    def observables(self) -> dict[str, "Observable"]:
        return self._observables

    def get_observable(self, name: str) -> "Observable":
        if name not in self._observables:
            raise ModelCapabilityError(
                f"Model '{self.name}' has no observable '{name}'; "
                f"available: {sorted(self._observables)}"
            )
        return self._observables[name]

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
