"""Simulation-method abstraction for QuaPh.

Each simulation technique (analytic exact diagonalization, VQE, IQPE, DMRG) is a
:class:`SimulationMethod` subclass living in its own module. A method declares its
tunable parameters via :class:`ParamSpec` (the single source of truth for both
Python-dict validation and CLI flag generation) and implements the per-cell
compute hooks. The orchestrator in :mod:`quaph._run` owns everything shared:
sweep resolution, the cell grid, parallelism, task distribution, logging, and
plotting.

Error mitigation is handled by the :mod:`quaph._mitigation` framework
(:class:`~quaph._mitigation.MitigationConfig` and
:class:`~quaph._mitigation.MitigationStrategy`), which each method defines
independently via ``MITIGATION_CLASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from quaph._mitigation import MitigationConfig


class Method(Enum):
    ANALYTIC = "analytic"
    VQE = "vqe"
    IQPE = "iqpe"
    DMRG = "dmrg"

    @classmethod
    def coerce(cls, value) -> "Method":
        if isinstance(value, Method):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            for m in cls:
                if m.value == key or m.name.lower() == key:
                    return m
        raise ValueError(
            f"Unknown simulation method {value!r}; choose from "
            f"{[m.value for m in cls]}."
        )


# Canonical ordering used whenever methods are deduplicated or laid out.
METHOD_ORDER = (Method.ANALYTIC, Method.VQE, Method.IQPE, Method.DMRG)


@dataclass
class ParamSpec:
    """Declares one tunable parameter of a simulation method.

    ``name`` is the key used in the Python ``method_params`` dict. ``cli`` toggles
    whether the CLI auto-generates a ``--<method>-<name>`` flag for it; complex
    dict-valued parameters (ansatz/optimizer/initial_state/mitigation) set
    ``cli=False`` and are wired up with bespoke flags in :mod:`quaph._cli`.
    """

    name: str
    type: Callable[[str], Any]
    default: Any = None
    help: str = ""
    choices: tuple | None = None
    cli: bool = True
    metavar: str | None = None
    is_flag: bool = False  # store_true / BooleanOptionalAction style


@dataclass
class CellContext:
    """Per-cell scratch passed to every ``compute_*`` call."""

    ix: int
    iy: int
    cell_index: int
    n_sites: int = 0
    spin: int = 1
    n_orbitals: int = 0
    mapper: Any = None
    raw_dir: str | None = None
    tmp_dir: str | None = None
    label: str = ""


class SimulationMethod:
    """Base class for a simulation technique.

    Subclasses set the class attributes and implement the compute hooks they
    support. Unsupported hooks raise :class:`NotImplementedError`; the
    orchestrator gates on the ``SUPPORTS_*`` flags before ever calling them.

    Error mitigation is configured via the ``mitigation`` key in
    ``method_params``, which is validated against the method's
    ``MITIGATION_CLASS``.  Subclasses can override ``MITIGATION_CLASS`` to
    restrict or extend the available options for their technique.
    """

    METHOD: Method
    LABEL: str
    PARAM_SPECS: list[ParamSpec] = []
    EXTRA_PARAMS: tuple[str, ...] = ()
    MITIGATION_CLASS: type[MitigationConfig] = MitigationConfig
    SUPPORTS_REAL_SPACE: bool = True
    SUPPORTS_BAND_STRUCTURE: bool = False
    SUPPORTS_OPERATOR: bool = False
    WANTS_PARALLEL: bool = True

    def __init__(self, **params):
        raw_mitigation = params.pop("mitigation", None)
        self.mitigation: MitigationConfig = self.MITIGATION_CLASS.coerce(raw_mitigation)
        self.mitigation_strategies: list = []

        allowed = {s.name for s in self.PARAM_SPECS} | set(self.EXTRA_PARAMS)
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"Unknown {self.METHOD.value} parameter(s) {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}."
            )
        self.params: dict[str, Any] = {}
        for spec in self.PARAM_SPECS:
            self.params[spec.name] = params.get(spec.name, spec.default)
        for extra in self.EXTRA_PARAMS:
            self.params[extra] = params.get(extra, None)
        for key, value in self.params.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        return self.METHOD.value

    # ------------------------------------------------------ mitigation setup
    def calibrate_mitigation(self, backend) -> None:
        """Build and calibrate every active mitigation strategy.

        Call this **once in the main process** before dispatching parallel
        jobs. Strategies store only picklable calibration data on
        themselves (e.g. M3's confusion matrices, not a live mitigator
        object), so joblib can pickle this method instance — mitigation
        strategies included — across worker processes.
        """
        self.mitigation_strategies = self.mitigation.build_strategies()
        for strategy in self.mitigation_strategies:
            strategy.calibrate(backend)

    # ------------------------------------------------------------------ compute
    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx: CellContext) -> dict:
        raise NotImplementedError(
            f"{self.METHOD.value} does not support real-space runs."
        )

    def compute_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                           backend, ctx: CellContext) -> dict:
        raise NotImplementedError(
            f"{self.METHOD.value} does not support band-structure runs."
        )

    def compute_operator_cell(self, op, *, extremum, backend, label, observable: str = "E") -> dict:
        raise NotImplementedError(
            f"{self.METHOD.value} does not support the qubit-operator path."
        )

    # ------------------------------------------------------------------- reduce
    def reduce(self, cell: dict, *, extremum: str = "min", analytic: float | None = None):
        """Collapse a raw cell dict to the scalar (or band list) plotted/stored.

        When `analytic` is given, pick whichever repetition lands closest to
        it rather than blindly taking min/max. Blind min/max biases the 
        result toward whichever noisy repetition happened to swing
        furthest in the "good" direction (most negative for extremum="min"),
        which isn't necessarily the most representative one and can distort
        raw vs mitigated comparisons that pair repetitions across runs.
        Falls back to blind min/max when no analytic reference is available
        """
        if "value" in cell:
            return cell["value"]
        if "bands" in cell:
            return cell["bands"]
        reps = cell.get("repetitions") or []
        if not reps:
            return float("nan")
        if analytic is not None:
            return float(min(reps, key=lambda r: abs(r - analytic)))
        return float(max(reps) if extremum == "max" else min(reps))

    # -------------------------------------------------------------- diagnostics
    def parameter_summary(self) -> dict:
        """JSON-friendly snapshot of this method's parameters for the log file."""
        out = dict(self.params)
        if self.mitigation.any_active():
            out["mitigation"] = self.mitigation.summary()
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[Method, type[SimulationMethod]] = {}


def register_method(cls: type[SimulationMethod]) -> type[SimulationMethod]:
    METHOD_REGISTRY[cls.METHOD] = cls
    return cls


def _ensure_registered() -> None:
    import quaph._analytic  # noqa: F401
    import quaph._vqe  # noqa: F401
    import quaph._iqpe  # noqa: F401
    import quaph._dmrg  # noqa: F401


def build_method(method, params: dict | None = None) -> SimulationMethod:
    method = Method.coerce(method)
    _ensure_registered()
    cls = METHOD_REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"No implementation registered for method {method.value!r}.")
    return cls(**(params or {}))


def get_method_class(method) -> type[SimulationMethod]:
    method = Method.coerce(method)
    _ensure_registered()
    return METHOD_REGISTRY[method]