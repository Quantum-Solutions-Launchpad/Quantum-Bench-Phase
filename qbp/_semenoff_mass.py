"""Radial Semenoff-mass interface investigation for Haldane-like A/B lattices.

Replaces the uniform sublattice mass with a radial profile ``M(r)``: it adds
``M(r) - M`` to the A/B onsite terms, so the base model parameter ``M`` should
usually be ``0.0`` when the profile supplies the full mass. A sharp
(``radial_step``) or smoothed (``radial_tanh``) interface separates an inner mass
from an outer mass at ``radius``.

The mass-profile helpers below are specific to this investigation; the generic
radial/diagonal primitives they build on (``_radial_distances``,
``_add_site_diagonal``, ``_normalize_optional``) live in :mod:`qbp._profiles`.
"""

from __future__ import annotations

import numpy as np

from qbp._method import ParamSpec
from qbp._model import ModelCapabilityError
from qbp._investigation import Investigation, register_investigation
from qbp._profiles import _add_site_diagonal, _normalize_optional, _radial_distances


def normalize_mass_profile(profile: str | None) -> str:
    return _normalize_optional(
        profile,
        allowed={"none", "radial_step", "radial_tanh"},
        aliases={
            "off": "none",
            "false": "none",
            "step": "radial_step",
            "topological": "radial_step",
            "topological_interface": "radial_step",
            "smooth": "radial_tanh",
            "smooth_topological": "radial_tanh",
        },
        label="mass profile",
    )


def radial_mass_values(
    xy: np.ndarray,
    *,
    profile: str,
    radius: float,
    mass_inner: float,
    mass_outer: float,
    xi: float | None = None,
    center=None,
) -> np.ndarray:
    """Per-site target mass from a radial interface profile.

    Evaluates a mass that switches from ``mass_inner`` to ``mass_outer`` across
    ``radius`` at each site in ``xy`` (measured from ``center``). ``profile`` is
    ``'radial_step'`` for a sharp interface or ``'radial_tanh'`` for one
    smoothed over width ``xi``. This is the profile the :class:`SemenoffMass`
    investigation imposes on an A/B lattice.
    """
    mode = normalize_mass_profile(profile)
    if mode == "none":
        return np.zeros(len(xy), dtype=float)
    r0 = float(radius)
    r = _radial_distances(xy, center)
    m_in = float(mass_inner)
    m_out = float(mass_outer)
    if mode == "radial_step":
        return np.where(r <= r0, m_in, m_out).astype(float)
    width = max(float(xi if xi is not None else 1.0), 1e-12)
    step = 0.5 * (1.0 + np.tanh((r - r0) / width))
    return m_in + (m_out - m_in) * step


def _sublattice_signs(model, projection) -> np.ndarray:
    names = tuple(str(x).upper() for x in model.sublattices)
    if "A" not in names or "B" not in names:
        raise ValueError(
            "radial mass profiles currently require A/B sublattices, as in the Haldane model."
        )
    signs = np.zeros(len(projection.site_indices), dtype=float)
    for out_idx, site_idx in enumerate(projection.site_indices):
        sub_idx = int(site_idx % model.sites_per_cell)
        sub_name = str(model.sublattices[sub_idx]).upper()
        if sub_name == "A":
            signs[out_idx] = +1.0
        elif sub_name == "B":
            signs[out_idx] = -1.0
        else:
            raise ValueError(
                "radial mass profiles currently require every active site to belong to A or B."
            )
    return signs


@register_investigation
class SemenoffMass(Investigation):
    """Radial Semenoff-mass interface investigation for Haldane-like A/B lattices.

    Replaces the uniform sublattice mass with a radial profile :math:`M(r)`,
    adding :math:`M(r) - M` to the A/B onsite terms so an inner mass meets an
    outer mass across ``radius`` (sharp ``radial_step`` or smoothed
    ``radial_tanh``). Select it with ``investigation="semenoff_mass"`` and set
    the base model parameter ``M`` to ``0.0`` when the profile supplies the full
    mass.
    """

    NAME = "semenoff_mass"
    LABEL = "Semenoff mass $M(r)$"
    PARAM_SPECS = [
        ParamSpec("profile", str, default="radial_tanh",
                  choices=("radial_step", "radial_tanh"),
                  help="Radial mass interface shape (sharp step or smoothed tanh)."),
        ParamSpec("radius", float, help="Interface radius separating inner/outer mass."),
        ParamSpec("inner", float, help="Mass value inside the interface."),
        ParamSpec("outer", float, help="Mass value outside the interface."),
        ParamSpec("xi", float, help="Smoothing length, used by the radial_tanh profile."),
        ParamSpec("center", float, default=None, cli=False,
                  help="(x, y) profile center; defaults to the active geometry center."),
    ]

    def _validate(self) -> None:
        self.profile = normalize_mass_profile(self.profile)
        if self.profile == "none":
            raise ValueError(
                "SemenoffMass requires profile='radial_step' or 'radial_tanh'."
            )
        for field in ("radius", "inner", "outer"):
            if getattr(self, field) is None:
                raise ValueError(f"SemenoffMass requires '{field}'.")

    def check_model(self, model) -> None:
        names = tuple(str(x).upper() for x in model.sublattices)
        if "A" not in names or "B" not in names:
            raise ModelCapabilityError(
                f"investigation '{self.NAME}' requires A/B sublattices "
                f"(as in the Haldane model); model '{model.name}' has "
                f"sublattices {tuple(model.sublattices)}."
            )

    def apply(self, H: np.ndarray, model, projection, model_params: dict) -> np.ndarray:
        if "M" not in model_params:
            raise ValueError(
                "investigation 'semenoff_mass' requires the base model parameter 'M'."
            )
        target_mass = radial_mass_values(
            projection.positions,
            profile=self.profile,
            radius=self.radius,
            mass_inner=self.inner,
            mass_outer=self.outer,
            xi=self.xi,
            center=self.center,
        )
        delta_mass = target_mass - float(model_params["M"])
        return _add_site_diagonal(H, model.spin, delta_mass * _sublattice_signs(model, projection))
