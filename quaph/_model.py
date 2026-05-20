from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ModelCapabilityError(Exception):
    """Raised when a :class:`Model` is asked for a capability it does not implement.

    .. todo:: Document the situations that raise this (missing ``bloch_hamiltonian``,
       unknown observable, momentum-axis mismatch).
    """
    pass


@dataclass
class Observable:
    """A scalar (or per-band) quantity computed from a diagonalized Hamiltonian.

    .. todo:: Describe the observable contract, the difference between
       real-space (``analytic``) and momentum-space (``analytic_bloch``)
       callbacks, and how custom observables are attached to a :class:`Model`.

    Parameters
    ----------
    name : str
        Short identifier used as the lookup key in ``Model.observables``.
    display_name : str
        LaTeX-formatted label used on plot axes.
    analytic : callable
        Real-space evaluator with signature
        ``(model, lattice, H, eigvals, eigvecs, n_occ, params) -> float``.
    analytic_bloch : callable, optional
        Momentum-space evaluator with signature
        ``(model, k_tuple, H, eigvals, eigvecs, params) -> float | list[float]``.
    operator : Any, optional
        Optional Qiskit operator representation (reserved for future use).
    """
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


def _default_gap_analytic(model, lattice, H, eigvals, eigvecs, n_occ, params):
    if n_occ <= 0 or n_occ >= len(eigvals):
        return 0.0
    return float(eigvals[n_occ] - eigvals[n_occ - 1])


def default_gap_observable() -> "Observable":
    return Observable(
        name="gap",
        display_name=r"\Delta_{\mathrm{gap}}",
        analytic=_default_gap_analytic,
    )


def _default_kinetic_analytic(model, lattice, H, eigvals, eigvecs, n_occ, params):
    return float(__import__("numpy").sum(eigvals[:n_occ]))


def default_kinetic_observable() -> "Observable":
    return Observable(
        name="kinetic_energy",
        display_name=r"E_{\mathrm{kin}}",
        analytic=_default_kinetic_analytic,
    )


def _default_interaction_analytic(model, lattice, H, eigvals, eigvecs, n_occ, params):
    if model._mean_field_correction_fn is None:
        return 0.0
    return float(model._mean_field_correction_fn(lattice, n_occ, **params))


def default_interaction_observable() -> "Observable":
    return Observable(
        name="interaction_energy",
        display_name=r"E_{\mathrm{int}}",
        analytic=_default_interaction_analytic,
    )


def _default_density_variance_analytic(model, lattice, H, eigvals, eigvecs, n_occ, params):
    import numpy as np
    if n_occ <= 0:
        return 0.0
    V_occ = eigvecs[:, :n_occ]
    rho_diag = np.real(np.einsum("ij,ij->i", V_occ.conj(), V_occ))
    return float(np.var(rho_diag))


def default_density_variance_observable() -> "Observable":
    return Observable(
        name="density_variance",
        display_name=r"\mathrm{Var}(\langle n_i \rangle)",
        analytic=_default_density_variance_analytic,
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
    """A tight-binding lattice model with classical and quantum simulation support.

    .. todo:: Describe the role of :class:`Model` in QuaPh: how it bundles a
       real-space Hamiltonian, an optional Bloch Hamiltonian, an optimizer /
       mapper / ansatz triple, and observables into a single object that the
       runners (:func:`quaph.run_analytic`, :func:`quaph.run_simulated_ideal`,
       :func:`quaph.run_simulated_noisy`) can drive. Cover when to provide
       ``hamiltonian_matrix`` vs. ``bloch_hamiltonian`` vs. both.

    Parameters
    ----------
    name : str
        Unique registry key (e.g. ``"ssh"``).
    display_name : str
        Human-readable name used in plots and the console UI.
    param_labels : dict[str, str]
        Map of model-parameter names to LaTeX labels (without ``$``).
    spin : {1, 2}
        Spinless (1) or spinful (2).
    n_dims : {1, 2, 3}
        Spatial dimensionality of the lattice.
    lattice_shape : tuple[str, ...]
        Names of the lattice extents (e.g. ``("Lx", "Ly")``); length must match ``n_dims``.
    sites_per_cell : int
        Number of sublattice sites per unit cell.
    hamiltonian_matrix : callable
        Real-space builder ``(lattice, **params) -> ndarray`` returning the
        single-particle Hamiltonian.
    interaction_hamiltonian : callable, optional
        Builder ``(lattice, **params) -> FermionicOp`` for many-body terms.
    get_optimizer : callable, optional
        Factory ``(max_iters) -> Optimizer`` for VQE.
    get_mapper : callable, optional
        Factory ``(n_sites, spin, n_occ) -> QubitMapper``.
    get_vqe_ansatz : callable, optional
        Factory ``(n_qubits, n_layers, n_occ, spin) -> QuantumCircuit``.
    mean_field_correction : callable, optional
        Function ``(lattice, n_occ, **params) -> float`` adding a mean-field
        energy contribution to analytic results.
    bloch_hamiltonian : callable, optional
        Momentum-space builder ``(*ks, **params) -> ndarray``; required for
        band-structure runs.
    observables : dict[str, Observable], optional
        Extra observables to merge with the built-in defaults
        (``E``, ``gap``, ``kinetic_energy``, ``interaction_energy``,
        ``density_variance``).

    Raises
    ------
    ValueError
        If ``hamiltonian_matrix`` is missing, ``spin`` is not in ``{1, 2}``,
        ``n_dims`` is not in ``{1, 2, 3}``, ``lattice_shape`` length does not
        match ``n_dims``, or ``sites_per_cell`` is not a positive int.
    """
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

        merged_observables: dict[str, Observable] = {
            "E": default_energy_observable(),
            "gap": default_gap_observable(),
            "kinetic_energy": default_kinetic_observable(),
            "interaction_energy": default_interaction_observable(),
            "density_variance": default_density_variance_observable(),
        }
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
        """Return the :class:`Observable` registered under ``name``.

        .. todo:: Describe the lookup semantics and the default observables
           that are always available.

        Parameters
        ----------
        name : str
            Observable key (e.g. ``"E"``, ``"gap"``).

        Returns
        -------
        Observable
            The matching observable.

        Raises
        ------
        ModelCapabilityError
            If ``name`` is not a registered observable on this model.
        """
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
