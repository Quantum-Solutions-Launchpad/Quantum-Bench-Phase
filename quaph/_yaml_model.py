from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
import yaml
from asteval import Interpreter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quaph._model import Model, Observable, default_energy_observable


_QISKIT_OPTIMIZERS = (
    "SPSA", "COBYLA", "L_BFGS_B", "NELDER_MEAD", "POWELL", "SLSQP",
    "NFT", "ADAM", "GradientDescent", "QNSPSA", "AQGD", "CG", "P_BFGS",
)

_QISKIT_MAPPERS = (
    "JordanWignerMapper", "ParityMapper", "BravyiKitaevMapper",
)

_QISKIT_ANSATZES = (
    "excitation_preserving", "efficient_su2", "real_amplitudes",
    "pauli_two_design", "n_local", "two_local",
)

_INITIAL_STATES = (
    "hartree_fock", "uniform", "computational_zero", "vqe_informed",
)


def _make_evaluator(extra_names: dict | None = None) -> Interpreter:
    aeval = Interpreter(use_numpy=True, minimal=False)
    for unsafe in (
        "open", "exec", "eval", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "input", "exit", "quit",
        "print", "help",
    ):
        aeval.symtable.pop(unsafe, None)
    if extra_names:
        aeval.symtable.update(extra_names)
    return aeval


def _eval_expr(expr: str, names: dict) -> complex:
    aeval = _make_evaluator(names)
    val = aeval(expr, show_errors=False)
    if aeval.error:
        msg = "; ".join(e.get_error()[1] for e in aeval.error)
        raise ValueError(f"failed to evaluate expression {expr!r}: {msg}")
    return val


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str


class OnsiteTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["onsite"]
    sublattice: str
    coefficient: str
    spin_channels: list[Literal["up", "down"]] | None = None


class HoppingTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["hopping"]
    from_: str = Field(alias="from")
    to: str
    offsets: list[list[int]]
    coefficient: str
    hermitian_partner: bool = False
    spin_channels: list[Literal["up", "down"]] | None = None


class DensityDensityTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["density_density"]
    sublattice: str | None = None
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    offsets: list[list[int]] | None = None
    coefficient: str

    @model_validator(mode="after")
    def _check_variant(self) -> "DensityDensityTerm":
        bond_fields_set = sum(
            x is not None for x in (self.from_, self.to, self.offsets)
        )
        if bond_fields_set == 0:
            return self
        if bond_fields_set != 3:
            raise ValueError(
                "density_density bond variant requires all of from/to/offsets; "
                "omit them (and optionally set 'sublattice') for the onsite variant"
            )
        if self.sublattice is not None:
            raise ValueError(
                "density_density: cannot combine 'sublattice' (onsite) with from/to/offsets (bond)"
            )
        return self

    @property
    def is_onsite(self) -> bool:
        return self.from_ is None


class OptimizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in _QISKIT_OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer '{v}'; supported: {', '.join(_QISKIT_OPTIMIZERS)}"
            )
        return v


class MapperSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in _QISKIT_MAPPERS:
            raise ValueError(
                f"unknown mapper '{v}'; supported: {', '.join(_QISKIT_MAPPERS)}"
            )
        return v


class AnsatzSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    initial_state_prefix: Literal["hartree_fock", "none"] = "hartree_fock"

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in _QISKIT_ANSATZES:
            raise ValueError(
                f"unknown ansatz '{v}'; supported: {', '.join(_QISKIT_ANSATZES)}"
            )
        return v


class InitialStateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["hartree_fock", "uniform", "computational_zero", "vqe_informed"]
    vqe_ansatz: AnsatzSpec | None = None
    vqe_n_layers: int = 1
    vqe_max_iters: int = 100
    vqe_optimizer: OptimizerSpec | None = None


class ObservableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str
    analytic: str
    analytic_bloch: str | None = None


class BlochSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape: list[int]
    let: dict[str, Any] = Field(default_factory=dict)
    entries: dict[str, str]


HamiltonianTerm = Annotated[
    Union[OnsiteTerm, HoppingTerm, DensityDensityTerm],
    Field(discriminator="kind"),
]


class YamlModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    display_name: str
    spin: Literal[1, 2]
    n_dims: Literal[1, 2, 3]
    lattice_shape: list[str]
    sites_per_cell: int = Field(ge=1)
    sublattices: list[str]
    parameters: dict[str, ParameterSpec]
    terms: list[HamiltonianTerm] = Field(default_factory=list)
    optimizer: OptimizerSpec | None = None
    mapper: MapperSpec | None = None
    ansatz: AnsatzSpec | None = None
    iqpe_initial_state: InitialStateSpec | None = None
    bloch_hamiltonian: BlochSpec | None = None
    lattice_vectors: list[list[float]] | None = None
    sublattice_positions: dict[str, list[float]] | None = None
    observables: dict[str, ObservableSpec] | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "YamlModelSpec":
        if len(self.lattice_shape) != self.n_dims:
            raise ValueError(
                f"lattice_shape has {len(self.lattice_shape)} entries but n_dims={self.n_dims}"
            )
        if len(self.sublattices) != self.sites_per_cell:
            raise ValueError(
                f"sublattices has {len(self.sublattices)} entries but sites_per_cell={self.sites_per_cell}"
            )
        known_sub = set(self.sublattices)
        for t in self.terms:
            if isinstance(t, OnsiteTerm):
                if t.sublattice not in known_sub:
                    raise ValueError(f"onsite term references unknown sublattice '{t.sublattice}'")
                if t.spin_channels is not None and self.spin != 2:
                    raise ValueError(f"{t.kind} term has spin_channels but model spin={self.spin}")
            elif isinstance(t, HoppingTerm):
                if t.from_ not in known_sub:
                    raise ValueError(f"hopping term 'from' references unknown sublattice '{t.from_}'")
                if t.to not in known_sub:
                    raise ValueError(f"hopping term 'to' references unknown sublattice '{t.to}'")
                for off in t.offsets:
                    if len(off) != self.n_dims:
                        raise ValueError(
                            f"hopping offset {off} has length {len(off)} but n_dims={self.n_dims}"
                        )
                if t.spin_channels is not None and self.spin != 2:
                    raise ValueError(f"{t.kind} term has spin_channels but model spin={self.spin}")
            elif isinstance(t, DensityDensityTerm):
                if t.is_onsite:
                    if self.spin != 2:
                        raise ValueError("density_density onsite variant requires spin=2")
                    if t.sublattice is not None and t.sublattice not in known_sub:
                        raise ValueError(
                            f"density_density 'sublattice' references unknown sublattice '{t.sublattice}'"
                        )
                else:
                    if t.from_ not in known_sub:
                        raise ValueError(
                            f"density_density 'from' references unknown sublattice '{t.from_}'"
                        )
                    if t.to not in known_sub:
                        raise ValueError(
                            f"density_density 'to' references unknown sublattice '{t.to}'"
                        )
                    for off in t.offsets:
                        if len(off) != self.n_dims:
                            raise ValueError(
                                f"density_density offset {off} has length {len(off)} but n_dims={self.n_dims}"
                            )
        if self.bloch_hamiltonian is not None:
            n = self.sites_per_cell
            if self.bloch_hamiltonian.shape != [n, n]:
                raise ValueError(
                    f"bloch_hamiltonian.shape must be [{n}, {n}] (= sites_per_cell × sites_per_cell)"
                )
            for k in self.bloch_hamiltonian.entries:
                parts = [p.strip() for p in k.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"bloch entry key '{k}' must be 'row,col'")
                try:
                    r, c = int(parts[0]), int(parts[1])
                except ValueError:
                    raise ValueError(f"bloch entry key '{k}' must be integer 'row,col'")
                if not (0 <= r < n and 0 <= c < n):
                    raise ValueError(f"bloch entry key '{k}' out of range for shape [{n},{n}]")
        if self.lattice_vectors is not None:
            if len(self.lattice_vectors) != self.n_dims:
                raise ValueError(
                    f"lattice_vectors has {len(self.lattice_vectors)} entries but n_dims={self.n_dims}"
                )
            for v in self.lattice_vectors:
                if len(v) != self.n_dims:
                    raise ValueError(
                        f"lattice vector {v} has length {len(v)} but n_dims={self.n_dims}"
                    )
        if self.sublattice_positions is not None:
            if self.lattice_vectors is None:
                raise ValueError("sublattice_positions requires lattice_vectors")
            for name, pos in self.sublattice_positions.items():
                if name not in known_sub:
                    raise ValueError(f"sublattice_positions references unknown sublattice '{name}'")
                if len(pos) != self.n_dims:
                    raise ValueError(
                        f"sublattice position for '{name}' has length {len(pos)} but n_dims={self.n_dims}"
                    )
        return self


def _flat_cell_index(cell_idx: tuple[int, ...], lattice: tuple[int, ...]) -> int:
    flat = 0
    for c, L in zip(cell_idx, lattice):
        flat = flat * L + c
    return flat


def _iter_cells(lattice: tuple[int, ...]):
    if len(lattice) == 1:
        for i in range(lattice[0]):
            yield (i,)
    elif len(lattice) == 2:
        for i in range(lattice[0]):
            for j in range(lattice[1]):
                yield (i, j)
    elif len(lattice) == 3:
        for i in range(lattice[0]):
            for j in range(lattice[1]):
                for k in range(lattice[2]):
                    yield (i, j, k)


def _build_hamiltonian_matrix(spec: YamlModelSpec):
    sub_index = {name: i for i, name in enumerate(spec.sublattices)}
    sites_per_cell = spec.sites_per_cell
    spin = spec.spin
    param_names = list(spec.parameters)

    def hamiltonian_matrix(lattice, **params):
        for pn in param_names:
            if pn not in params:
                raise TypeError(f"missing parameter '{pn}'")
        lat = tuple(lattice)
        if len(lat) != spec.n_dims:
            raise ValueError(f"lattice has {len(lat)} dims, model expects {spec.n_dims}")
        n_cells = 1
        for L in lat:
            n_cells *= L
        n_sites = n_cells * sites_per_cell
        dim = n_sites * spin
        H = np.zeros((dim, dim), dtype=complex)

        def orbital(cell: tuple[int, ...], sub: int, s: int) -> int:
            return (_flat_cell_index(cell, lat) * sites_per_cell + sub) * spin + s

        for term in spec.terms:
            if isinstance(term, DensityDensityTerm):
                continue
            if isinstance(term, OnsiteTerm):
                coef = complex(_eval_expr(term.coefficient, params))
                sub = sub_index[term.sublattice]
                channels = _resolve_spin_channels(term.spin_channels, spin)
                for cell in _iter_cells(lat):
                    for s in channels:
                        o = orbital(cell, sub, s)
                        H[o, o] += coef
            elif isinstance(term, HoppingTerm):
                coef = complex(_eval_expr(term.coefficient, params))
                src = sub_index[term.from_]
                dst = sub_index[term.to]
                channels = _resolve_spin_channels(term.spin_channels, spin)
                for cell in _iter_cells(lat):
                    for off in term.offsets:
                        tgt_cell = tuple((c + o) % L for c, o, L in zip(cell, off, lat))
                        for s in channels:
                            i_orb = orbital(cell, src, s)
                            j_orb = orbital(tgt_cell, dst, s)
                            H[i_orb, j_orb] += coef
                            if term.hermitian_partner:
                                H[j_orb, i_orb] += np.conj(coef)
        return H

    hamiltonian_matrix.__name__ = f"_yaml_H_{spec.name.replace('-', '_')}"
    return hamiltonian_matrix


def _resolve_spin_channels(spec_channels: list[str] | None, spin: int) -> list[int]:
    if spin == 1:
        return [0]
    if spec_channels is None:
        return list(range(spin))
    mapping = {"up": 0, "down": 1}
    return [mapping[c] for c in spec_channels]


def _dd_terms(spec: YamlModelSpec) -> list[DensityDensityTerm]:
    return [t for t in spec.terms if isinstance(t, DensityDensityTerm)]


def _build_interaction_hamiltonian(spec: YamlModelSpec):
    dd = _dd_terms(spec)
    if not dd:
        return None
    from qiskit_nature.second_q.operators import FermionicOp
    sites_per_cell = spec.sites_per_cell
    spin = spec.spin
    sub_index = {name: i for i, name in enumerate(spec.sublattices)}

    def interaction(lattice, **params):
        lat = tuple(lattice)
        n_cells = 1
        for L in lat:
            n_cells *= L
        n_sites = n_cells * sites_per_cell
        num_so = n_sites * spin
        op = 0.0 * FermionicOp({}, num_spin_orbitals=num_so)

        def orbital(cell, sub, s):
            return (_flat_cell_index(cell, lat) * sites_per_cell + sub) * spin + s

        for term in dd:
            coef = complex(_eval_expr(term.coefficient, params))
            if coef == 0:
                continue
            if term.is_onsite:
                target_subs = (list(range(sites_per_cell))
                               if term.sublattice is None
                               else [sub_index[term.sublattice]])
                for cell in _iter_cells(lat):
                    for sub in target_subs:
                        up = orbital(cell, sub, 0)
                        dn = orbital(cell, sub, 1)
                        op += FermionicOp(
                            {f"+_{up} -_{up} +_{dn} -_{dn}": coef},
                            num_spin_orbitals=num_so,
                        )
            else:
                src = sub_index[term.from_]
                dst = sub_index[term.to]
                for cell in _iter_cells(lat):
                    for off in term.offsets:
                        tgt_cell = tuple((c + o) % L for c, o, L in zip(cell, off, lat))
                        for s_a in range(spin):
                            i_orb = orbital(cell, src, s_a)
                            for s_b in range(spin):
                                j_orb = orbital(tgt_cell, dst, s_b)
                                if i_orb == j_orb:
                                    continue
                                op += FermionicOp(
                                    {f"+_{i_orb} -_{i_orb} +_{j_orb} -_{j_orb}": coef},
                                    num_spin_orbitals=num_so,
                                )
        return op

    interaction.__name__ = f"_yaml_int_{spec.name.replace('-', '_')}"
    return interaction


def _build_mean_field_correction(spec: YamlModelSpec):
    dd = _dd_terms(spec)
    if not dd:
        return None
    sites_per_cell = spec.sites_per_cell
    spin = spec.spin

    def mean_field_correction(lattice, n_occ, **params):
        lat = tuple(lattice)
        n_cells = 1
        for L in lat:
            n_cells *= L
        n_sites = n_cells * sites_per_cell
        rho = n_occ / (spin * n_sites)
        total = 0.0
        for term in dd:
            coef = float(np.real(_eval_expr(term.coefficient, params)))
            if coef == 0.0:
                continue
            if term.is_onsite:
                n_target = n_sites if term.sublattice is None else n_cells
                total += coef * n_target * rho * rho
            else:
                n_bonds_dir = n_cells * len(term.offsets)
                total += coef * n_bonds_dir * (spin * spin) * rho * rho
        return float(total)

    mean_field_correction.__name__ = f"_yaml_mf_{spec.name.replace('-', '_')}"
    return mean_field_correction


def _build_optimizer_factory(spec: YamlModelSpec):
    if spec.optimizer is None:
        return None
    return build_optimizer_factory(spec.optimizer, name=spec.name)


def build_optimizer_factory(optimizer_spec: OptimizerSpec, *, name: str):
    import qiskit_algorithms.optimizers as qopt
    cls = getattr(qopt, optimizer_spec.type, None)
    if cls is None:
        raise ValueError(
            f"qiskit_algorithms.optimizers has no '{optimizer_spec.type}'"
        )
    raw_kwargs = dict(optimizer_spec.kwargs)

    def get_optimizer(max_iters):
        runtime = {"max_iters": max_iters}
        resolved = _resolve_runtime_kwargs(raw_kwargs, runtime, context="optimizer")
        return cls(**resolved)

    get_optimizer.__name__ = f"_opt_{name.replace('-', '_')}"
    return get_optimizer


def _resolve_runtime_kwargs(raw_kwargs: dict, runtime: dict, *, context: str) -> dict:
    resolved = {}
    for k, v in raw_kwargs.items():
        if isinstance(v, str) and v.startswith("@"):
            key = v[1:]
            if key not in runtime:
                raise ValueError(
                    f"{context} kwarg '{k}' references unknown runtime arg '{key}'; "
                    f"available: {sorted(runtime)}"
                )
            resolved[k] = runtime[key]
        else:
            resolved[k] = v
    return resolved


def _build_mapper_factory(spec: YamlModelSpec):
    if spec.mapper is None:
        return None
    import qiskit_nature.second_q.mappers as qmappers
    cls = getattr(qmappers, spec.mapper.type, None)
    if cls is None:
        raise ValueError(
            f"qiskit_nature.second_q.mappers has no '{spec.mapper.type}'"
        )
    raw_kwargs = dict(spec.mapper.kwargs)

    def get_mapper(n_sites: int, spin: int, n_occ: int):
        num_particles = (n_occ // 2 + n_occ % 2, n_occ // 2) if spin == 2 else (n_occ, 0)
        runtime = {
            "n_sites": n_sites,
            "spin": spin,
            "n_occ": n_occ,
            "num_particles": num_particles,
        }
        resolved = _resolve_runtime_kwargs(raw_kwargs, runtime, context="mapper")
        return cls(**resolved)

    get_mapper.__name__ = f"_yaml_mapper_{spec.name.replace('-', '_')}"
    return get_mapper


def _build_ansatz_factory(spec: YamlModelSpec):
    if spec.ansatz is None:
        return None
    return build_ansatz_factory(spec.ansatz, name=spec.name)


def build_ansatz_factory(ansatz_spec: AnsatzSpec, *, name: str):
    import qiskit.circuit.library as qlib
    from qiskit import QuantumCircuit
    func = getattr(qlib, ansatz_spec.type, None)
    if func is None:
        raise ValueError(
            f"qiskit.circuit.library has no '{ansatz_spec.type}'"
        )
    raw_kwargs = dict(ansatz_spec.kwargs)
    prefix_kind = ansatz_spec.initial_state_prefix

    def get_vqe_ansatz(n_qubits: int, n_layers: int, n_occ: int, spin: int):
        n_sites = n_qubits // spin if spin else n_qubits
        runtime = {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "n_occ": n_occ,
            "spin": spin,
            "n_sites": n_sites,
        }
        resolved = _resolve_runtime_kwargs(raw_kwargs, runtime, context="ansatz")
        body = func(n_qubits, **resolved)
        if prefix_kind == "hartree_fock":
            qc = QuantumCircuit(n_qubits)
            for i in range(n_occ):
                qc.x(i)
            qc.compose(body, inplace=True)
            return qc
        return body

    get_vqe_ansatz.__name__ = f"_ansatz_{name.replace('-', '_')}"
    return get_vqe_ansatz


def build_initial_state_factory(spec: InitialStateSpec, *, name: str):
    from quaph._core import _hf_initial_state, _uniform_initial, _zero_initial
    from quaph._vqe import _vqe_initial_state

    kind = spec.type

    if kind == "vqe_informed":
        vqe_ansatz_spec = spec.vqe_ansatz or AnsatzSpec(
            type="efficient_su2", kwargs={"reps": "@n_layers"}, initial_state_prefix="hartree_fock",
        )
        vqe_optimizer_spec = spec.vqe_optimizer or OptimizerSpec(
            type="SPSA", kwargs={"maxiter": "@max_iters"},
        )
        vqe_n_layers = spec.vqe_n_layers
        vqe_max_iters = spec.vqe_max_iters
        get_ansatz = build_ansatz_factory(vqe_ansatz_spec, name=f"{name}_vqe_init")
        get_optimizer = build_optimizer_factory(vqe_optimizer_spec, name=f"{name}_vqe_init")

    def get_initial_state(hamiltonian, *, n_occ: int = 0, spin: int = 1, mapper=None):
        n_qubits = hamiltonian.num_qubits
        if kind == "hartree_fock":
            n_sites = n_qubits // spin if spin else n_qubits
            return _hf_initial_state(n_sites, spin, n_occ, mapper)
        if kind == "uniform":
            return _uniform_initial(n_qubits)
        if kind == "computational_zero":
            return _zero_initial(n_qubits)
        # vqe_informed
        ansatz = get_ansatz(n_qubits, vqe_n_layers, n_occ, spin)
        return _vqe_initial_state(hamiltonian, ansatz, get_optimizer, vqe_max_iters)

    get_initial_state.__name__ = f"_initial_state_{name.replace('-', '_')}"
    return get_initial_state


def _build_initial_state_factory(spec: YamlModelSpec):
    if spec.iqpe_initial_state is None:
        return None
    return build_initial_state_factory(spec.iqpe_initial_state, name=spec.name)


def _build_bloch_hamiltonian(spec: YamlModelSpec):
    if spec.bloch_hamiltonian is None:
        return None
    bspec = spec.bloch_hamiltonian
    n = spec.sites_per_cell
    momentum_names = {1: ("k",), 2: ("kx", "ky"), 3: ("kx", "ky", "kz")}[spec.n_dims]

    let_items = list(bspec.let.items())
    entries = bspec.entries

    def bloch_hamiltonian(*ks, **params):
        if len(ks) != len(momentum_names):
            raise TypeError(f"expected {len(momentum_names)} momentum args, got {len(ks)}")
        names: dict[str, Any] = dict(params)
        for mn, val in zip(momentum_names, ks):
            names[mn] = val
        for let_name, raw in let_items:
            if isinstance(raw, str):
                names[let_name] = _eval_expr(raw, names)
            else:
                names[let_name] = raw
        H = np.zeros((n, n), dtype=complex)
        for key, expr in entries.items():
            r, c = (int(x.strip()) for x in key.split(","))
            H[r, c] = complex(_eval_expr(expr, names))
        return H

    bloch_hamiltonian.__name__ = f"_yaml_bloch_{spec.name.replace('-', '_')}"
    return bloch_hamiltonian


def _build_auto_bloch_hamiltonian(spec: YamlModelSpec):
    sub_index = {name: i for i, name in enumerate(spec.sublattices)}
    n = spec.sites_per_cell
    momentum_names = {1: ("k",), 2: ("kx", "ky"), 3: ("kx", "ky", "kz")}[spec.n_dims]
    param_names = list(spec.parameters)

    for t in spec.terms:
        if isinstance(t, DensityDensityTerm):
            continue
        if t.spin_channels is not None:
            return None

    use_cartesian = spec.lattice_vectors is not None
    if use_cartesian:
        a_mat = np.array(spec.lattice_vectors, dtype=float)
        positions = {name: np.zeros(spec.n_dims) for name in spec.sublattices}
        if spec.sublattice_positions is not None:
            for name, pos in spec.sublattice_positions.items():
                positions[name] = np.array(pos, dtype=float)

    def bloch_hamiltonian(*ks, **params):
        if len(ks) != len(momentum_names):
            raise TypeError(
                f"expected {len(momentum_names)} momentum args, got {len(ks)}"
            )
        for pn in param_names:
            if pn not in params:
                raise TypeError(f"missing parameter '{pn}'")
        H = np.zeros((n, n), dtype=complex)
        k_vec = np.array(ks, dtype=float) if use_cartesian else None
        for term in spec.terms:
            if isinstance(term, DensityDensityTerm):
                continue
            coef = complex(_eval_expr(term.coefficient, params))
            if isinstance(term, OnsiteTerm):
                a = sub_index[term.sublattice]
                H[a, a] += coef
            elif isinstance(term, HoppingTerm):
                a = sub_index[term.from_]
                b = sub_index[term.to]
                for off in term.offsets:
                    if use_cartesian:
                        d_cart = np.array(off, dtype=float) @ a_mat
                        d_cart = d_cart + positions[term.to] - positions[term.from_]
                        phase = np.exp(1j * float(k_vec @ d_cart))
                    else:
                        phase = np.exp(1j * sum(k * o for k, o in zip(ks, off)))
                    H[a, b] += coef * phase
                    if term.hermitian_partner:
                        H[b, a] += np.conj(coef) * np.conj(phase)
        return H

    bloch_hamiltonian.__name__ = f"_yaml_auto_bloch_{spec.name.replace('-', '_')}"
    return bloch_hamiltonian


def _build_observables(spec: YamlModelSpec) -> dict[str, Observable] | None:
    if not spec.observables:
        return None
    momentum_names = {1: ("k",), 2: ("kx", "ky"), 3: ("kx", "ky", "kz")}[spec.n_dims]
    out: dict[str, Observable] = {}
    for obs_name, ospec in spec.observables.items():
        analytic_expr = ospec.analytic
        bloch_expr = ospec.analytic_bloch

        def _make_analytic(expr=analytic_expr, oname=obs_name):
            def analytic_fn(model, lattice, H, eigvals, eigvecs, n_occ, params):
                V_occ = eigvecs[:, :n_occ]
                rho = V_occ @ V_occ.conj().T
                names: dict[str, Any] = dict(params)
                names.update({
                    "eigvals": eigvals, "eigvecs": eigvecs, "H": H,
                    "rho": rho, "n_occ": n_occ,
                    "n_sites": H.shape[0],
                    "lattice": lattice,
                })
                val = _eval_expr(expr, names)
                return float(np.real(val))
            analytic_fn.__name__ = f"_yaml_obs_{spec.name.replace('-', '_')}_{oname}"
            return analytic_fn

        def _make_bloch(expr=bloch_expr, oname=obs_name):
            if expr is None:
                return None
            def analytic_bloch_fn(model, k_tuple, H, eigvals, eigvecs, params):
                names: dict[str, Any] = dict(params)
                for mn, val in zip(momentum_names, k_tuple):
                    names[mn] = val
                names.update({
                    "eigvals": eigvals, "eigvecs": eigvecs, "H": H,
                    "n_bands": H.shape[0],
                })
                val = _eval_expr(expr, names)
                if isinstance(val, np.ndarray):
                    return val.tolist()
                if isinstance(val, (list, tuple)):
                    return list(val)
                return float(np.real(val))
            analytic_bloch_fn.__name__ = f"_yaml_obs_bloch_{spec.name.replace('-', '_')}_{oname}"
            return analytic_bloch_fn

        out[obs_name] = Observable(
            name=obs_name,
            display_name=ospec.display_name,
            analytic=_make_analytic(),
            analytic_bloch=_make_bloch(),
        )
    return out


def load_yaml_model(path: str | Path) -> Model:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    spec = YamlModelSpec.model_validate(data)
    return spec_to_model(spec)


def build_tight_binding_model(
    *,
    name: str,
    display_name: str,
    spin: int,
    n_dims: int,
    lattice_shape,
    sites_per_cell: int,
    sublattices,
    parameters: dict,
    terms: list,
    optimizer: dict | None = None,
    mapper: dict | None = None,
    ansatz: dict | None = None,
    bloch_hamiltonian: dict | None = None,
    lattice_vectors=None,
    sublattice_positions: dict | None = None,
    observables: dict | None = None,
) -> Model:
    data = {
        "name": name,
        "display_name": display_name,
        "spin": spin,
        "n_dims": n_dims,
        "lattice_shape": list(lattice_shape),
        "sites_per_cell": sites_per_cell,
        "sublattices": list(sublattices),
        "parameters": parameters,
        "terms": terms,
    }
    if optimizer is not None:
        data["optimizer"] = optimizer
    if mapper is not None:
        data["mapper"] = mapper
    if ansatz is not None:
        data["ansatz"] = ansatz
    if bloch_hamiltonian is not None:
        data["bloch_hamiltonian"] = bloch_hamiltonian
    if lattice_vectors is not None:
        data["lattice_vectors"] = [list(v) for v in lattice_vectors]
    if sublattice_positions is not None:
        data["sublattice_positions"] = {k: list(v) for k, v in sublattice_positions.items()}
    if observables is not None:
        data["observables"] = observables
    spec = YamlModelSpec.model_validate(data)
    return spec_to_model(spec)


def _build_interacting_analytic_observables(spec: YamlModelSpec) -> dict[str, Observable] | None:
    if not _dd_terms(spec):
        return None

    import math

    sites_per_cell = spec.sites_per_cell

    def _mb_result_cached(model, lattice, n_occ, params):
        """Returns (eigvals, ground_state_vec_in_sector, sector_mask)."""
        key = (tuple(lattice), int(n_occ), tuple(sorted(params.items())))
        cache = getattr(model, "_mb_cache", None)
        if cache is None:
            cache = {}
            model._mb_cache = cache
        if key in cache:
            return cache[key]
        n_sites = math.prod(lattice) * model.sites_per_cell
        fermionic_op = model.fermionic_hamiltonian(lattice, **params)
        mapper = model.get_mapper(n_sites, model.spin, n_occ)
        qubit_op = mapper.map(fermionic_op)
        H_mb = qubit_op.to_matrix()
        n_qubits = qubit_op.num_qubits
        occ_counts = np.array([bin(i).count('1') for i in range(2**n_qubits)])
        mask = occ_counts == n_occ
        H_sector = H_mb[np.ix_(mask, mask)]
        evs, vecs = np.linalg.eigh(H_sector)
        result = (evs, vecs[:, 0], mask)
        cache[key] = result
        return result

    def _spin_sector_expectation(model, lattice, n_occ, psi, mask, *, stagger: bool):
        from qiskit_nature.second_q.operators import FermionicOp
        lat = tuple(lattice)
        n_sites = math.prod(lat) * model.sites_per_cell
        num_so = n_sites * model.spin
        terms_dict: dict[str, complex] = {}
        for s in range(n_sites):
            sign = (+1.0 if (s % sites_per_cell) % 2 == 0 else -1.0) if stagger else +1.0
            up, dn = s * 2, s * 2 + 1
            terms_dict[f"+_{up} -_{up}"] = terms_dict.get(f"+_{up} -_{up}", 0.0) + sign / n_sites
            terms_dict[f"+_{dn} -_{dn}"] = terms_dict.get(f"+_{dn} -_{dn}", 0.0) - sign / n_sites
        spin_op = FermionicOp(terms_dict, num_spin_orbitals=num_so)
        mapper = model.get_mapper(n_sites, model.spin, n_occ)
        qubit_op = mapper.map(spin_op)
        Op_sector = qubit_op.to_matrix()[np.ix_(mask, mask)]
        return float(abs(np.real(psi.conj() @ Op_sector @ psi)))

    def _spin_fermionic_op(model, lattice, *, stagger: bool):
        from qiskit_nature.second_q.operators import FermionicOp
        lat = tuple(lattice)
        n_sites = math.prod(lat) * model.sites_per_cell
        num_so = n_sites * model.spin
        terms_dict: dict[str, complex] = {}
        for s in range(n_sites):
            sign = (+1.0 if (s % sites_per_cell) % 2 == 0 else -1.0) if stagger else +1.0
            up, dn = s * 2, s * 2 + 1
            terms_dict[f"+_{up} -_{up}"] = terms_dict.get(f"+_{up} -_{up}", 0.0) + sign / n_sites
            terms_dict[f"+_{dn} -_{dn}"] = terms_dict.get(f"+_{dn} -_{dn}", 0.0) - sign / n_sites
        return FermionicOp(terms_dict, num_spin_orbitals=num_so)

    def _e_mb(model, lattice, H, eigvals, eigvecs, n_occ, params):
        evs, _, _ = _mb_result_cached(model, lattice, n_occ, params)
        return float(evs[0]) if len(evs) > 0 else float('nan')

    def _gap_mb(model, lattice, H, eigvals, eigvecs, n_occ, params):
        evs, _, _ = _mb_result_cached(model, lattice, n_occ, params)
        if len(evs) < 2:
            return 0.0
        return float(evs[1] - evs[0])

    def _m_stag(model, lattice, H, eigvals, eigvecs, n_occ, params):
        _, psi, mask = _mb_result_cached(model, lattice, n_occ, params)
        return _spin_sector_expectation(model, lattice, n_occ, psi, mask, stagger=True)

    def _m_total(model, lattice, H, eigvals, eigvecs, n_occ, params):
        _, psi, mask = _mb_result_cached(model, lattice, n_occ, params)
        return _spin_sector_expectation(model, lattice, n_occ, psi, mask, stagger=False)

    def _m_stag_quantum_op(model, lattice, **params):
        return _spin_fermionic_op(model, lattice, stagger=True)

    def _m_total_quantum_op(model, lattice, **params):
        return _spin_fermionic_op(model, lattice, stagger=False)

    return {
        "E": Observable(
            name="E", display_name="E",
            analytic=_e_mb,
            analytic_bloch=default_energy_observable().analytic_bloch,
        ),
        "gap": Observable(
            name="gap", display_name=r"\Delta_{\mathrm{gap}}",
            analytic=_gap_mb,
        ),
        "M_stag": Observable(
            name="M_stag", display_name=r"|m_{\mathrm{stag}}|",
            analytic=_m_stag,
            quantum_operator=_m_stag_quantum_op,
        ),
        "M_total": Observable(
            name="M_total", display_name=r"|m_{\mathrm{total}}|",
            analytic=_m_total,
            quantum_operator=_m_total_quantum_op,
        ),
    }


def spec_to_model(spec: YamlModelSpec) -> Model:
    param_labels = {k: v.label for k, v in spec.parameters.items()}
    observables = _build_observables(spec) or {}
    int_obs = _build_interacting_analytic_observables(spec)
    if int_obs:
        for k, v in int_obs.items():
            observables.setdefault(k, v)
    return Model(
        name=spec.name,
        display_name=spec.display_name,
        param_labels=param_labels,
        spin=spec.spin,
        n_dims=spec.n_dims,
        lattice_shape=tuple(spec.lattice_shape),
        sites_per_cell=spec.sites_per_cell,
        hamiltonian_matrix=_build_hamiltonian_matrix(spec),
        interaction_hamiltonian=_build_interaction_hamiltonian(spec),
        get_optimizer=_build_optimizer_factory(spec),
        get_mapper=_build_mapper_factory(spec),
        get_vqe_ansatz=_build_ansatz_factory(spec),
        get_iqpe_initial_state=_build_initial_state_factory(spec),
        mean_field_correction=_build_mean_field_correction(spec),
        bloch_hamiltonian=(
            _build_bloch_hamiltonian(spec)
            if spec.bloch_hamiltonian is not None
            else _build_auto_bloch_hamiltonian(spec)
        ),
        observables=observables or None,
    )
