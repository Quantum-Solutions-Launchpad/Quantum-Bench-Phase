"""Model-specific physics investigations for qbp.

An :class:`Investigation` is a pluggable, model-specific modification of the
single-particle Hamiltonian -- the analogue of :class:`~qbp._method.Method` for
the *physics* being probed rather than the *solver* being used. Where boundary
conditions and the open-boundary geometry/potential are generic (every model
understands them), an investigation captures a study that only makes sense for
certain models, e.g. a radial Semenoff-mass interface on a Haldane-like A/B
lattice.

This module holds only the framework. Each concrete investigation lives in its
own module (e.g. :mod:`qbp._semenoff_mass`), declares its tunable parameters via
:class:`~qbp._method.ParamSpec`, gates itself on model capability through
``check_model``, modifies the projected Hamiltonian in ``apply``, and registers
itself with :func:`register_investigation`. Adding a new study therefore costs no
change to :func:`qbp.run` -- callers select it with ``investigation=<name or
instance>`` and ``investigation_params=<dict>``, exactly mirroring the
``model`` / ``model_params`` and ``method`` / ``method_params`` pairings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qbp._method import ParamSpec


class Investigation:
    """Base class for a model-specific Hamiltonian investigation.

    Subclasses set the class attributes and implement ``apply`` (and usually
    ``check_model``/``_validate``). Parameters declared in ``PARAM_SPECS`` become
    constructor keywords, are validated against the spec, and are exposed both as
    attributes and through :meth:`parameter_summary`.
    """

    NAME: str
    LABEL: str
    PARAM_SPECS: list[ParamSpec] = []

    def __init__(self, **params):
        allowed = {s.name for s in self.PARAM_SPECS}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"Unknown {self.NAME} parameter(s) {sorted(unknown)}; "
                f"allowed: {sorted(allowed)}."
            )
        self.params: dict[str, Any] = {}
        for spec in self.PARAM_SPECS:
            value = params.get(spec.name, spec.default)
            self.params[spec.name] = value
            setattr(self, spec.name, value)
        self._validate()

    # --------------------------------------------------------------- overridable
    def _validate(self) -> None:
        """Validate/normalize parameters at construction time. Override as needed."""

    def check_model(self, model) -> None:
        """Raise :class:`ModelCapabilityError` if ``model`` can't support this study."""

    def apply(self, H: np.ndarray, model, projection, model_params: dict) -> np.ndarray:
        """Return ``H`` with this investigation's terms added."""
        raise NotImplementedError(
            f"investigation {self.NAME!r} does not implement apply()."
        )

    # -------------------------------------------------------------- diagnostics
    @property
    def name(self) -> str:
        return self.NAME

    def parameter_summary(self) -> dict:
        """JSON-friendly snapshot of this investigation's parameters."""
        out = {}
        for key, value in self.params.items():
            if isinstance(value, (list, tuple)):
                out[key] = list(value)
            else:
                out[key] = value
        return out

    def metadata(self) -> dict:
        """Block describing this investigation for the run log."""
        return {"name": self.NAME, "params": self.parameter_summary()}


# String-keyed registry: investigations are open-ended (third parties can add
# their own), so unlike Method there is no closed enum.
INVESTIGATION_REGISTRY: dict[str, type[Investigation]] = {}


def register_investigation(cls: type[Investigation]) -> type[Investigation]:
    INVESTIGATION_REGISTRY[cls.NAME] = cls
    return cls


def _ensure_registered() -> None:
    # Import the investigation modules so they register themselves. Imported
    # lazily (mirroring qbp._method) to keep this framework module dependency-free.
    import qbp._semenoff_mass  # noqa: F401


def get_investigation_class(name: str) -> type[Investigation]:
    _ensure_registered()
    key = str(name).strip().lower()
    cls = INVESTIGATION_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown investigation {name!r}; choose from "
            f"{sorted(INVESTIGATION_REGISTRY)}."
        )
    return cls


def build_investigation(investigation, params: dict | None = None) -> Investigation | None:
    """Resolve ``investigation`` (name, instance, or ``None``) into an instance.

    ``params`` mirrors ``method_params``: it is applied only when selecting an
    investigation by name. Passing it alongside a prebuilt instance, or without
    any investigation, is an error.
    """
    if investigation is None:
        if params:
            raise ValueError(
                "investigation_params were provided without investigation=..."
            )
        return None
    if isinstance(investigation, Investigation):
        if params:
            raise ValueError(
                "pass parameters via the Investigation instance or investigation_params, "
                "not both."
            )
        return investigation
    cls = get_investigation_class(investigation)
    return cls(**(params or {}))


def registered_investigations() -> dict[str, type[Investigation]]:
    """Return the registry of all known investigations (for CLI/help generation)."""
    _ensure_registered()
    return dict(INVESTIGATION_REGISTRY)
