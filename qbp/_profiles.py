from __future__ import annotations

import numpy as np


def _normalize_optional(value: str | None, *, allowed: set[str], aliases: dict[str, str], label: str) -> str:
    if value is None:
        return "none"
    mode = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mode = aliases.get(mode, mode)
    if mode not in allowed:
        opts = ", ".join(sorted(allowed))
        raise ValueError(f"unsupported {label} {value!r}; expected one of: {opts}.")
    return mode


def normalize_potential_profile(profile: str | None) -> str:
    return _normalize_optional(
        profile,
        allowed={"none", "soft_dot"},
        aliases={
            "off": "none",
            "false": "none",
            "soft": "soft_dot",
            "soft_confinement": "soft_dot",
            "radial_tanh": "soft_dot",
            "tanh": "soft_dot",
        },
        label="potential profile",
    )


def _profile_center(xy: np.ndarray, center) -> np.ndarray:
    if center is not None:
        arr = np.asarray(center, dtype=float)
        if arr.shape != (2,):
            raise ValueError(f"profile center must have two coordinates; got {center!r}.")
        return arr
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    return 0.5 * (lo + hi)


def _radial_distances(xy: np.ndarray, center) -> np.ndarray:
    c = _profile_center(xy, center)
    return np.linalg.norm(xy - c[None, :], axis=1)


def soft_dot_potential(
    xy: np.ndarray,
    *,
    radius: float,
    v0: float,
    xi: float,
    center=None,
) -> np.ndarray:
    r"""Smooth radial confinement potential evaluated at each site.

    Returns the soft-dot potential
    :math:`V(r) = \tfrac{v_0}{2}\,[1 + \tanh((r - \text{radius})/\xi)]` for the
    Cartesian site coordinates ``xy``, measured from ``center`` (defaulting to
    the flake center). ``v0`` sets the barrier height and ``xi`` the wall
    softness. Add it to a Hamiltonian's diagonal with
    :func:`apply_profiles_to_hamiltonian`.
    """
    r0 = float(radius)
    width = max(float(xi), 1e-12)
    r = _radial_distances(xy, center)
    return 0.5 * float(v0) * (1.0 + np.tanh((r - r0) / width))


def _add_site_diagonal(H: np.ndarray, spin: int, site_values: np.ndarray) -> np.ndarray:
    out = H.copy()
    for site, value in enumerate(site_values):
        for s in range(spin):
            orbital = site * spin + s
            out[orbital, orbital] += value
    return out


def apply_profiles_to_hamiltonian(
    H: np.ndarray,
    model,
    projection,
    model_params: dict,
    *,
    potential_profile: str | None = None,
    potential_radius: float | None = None,
    potential_v0: float | None = None,
    potential_xi: float | None = None,
    center=None,
) -> np.ndarray:
    """Add the open-boundary scalar confinement potential V(r) to the diagonal.

    Model-specific onsite studies (e.g. the radial Semenoff mass) are handled by
    :mod:`qbp._investigation`, not here.
    """
    potential_mode = normalize_potential_profile(potential_profile)
    if potential_mode == "none":
        return H
    if potential_radius is None or potential_v0 is None:
        raise ValueError("soft-dot potential requires potential_radius and potential_v0.")
    values = soft_dot_potential(
        projection.positions,
        radius=potential_radius,
        v0=potential_v0,
        xi=0.8 if potential_xi is None else potential_xi,
        center=center,
    )
    return _add_site_diagonal(H, model.spin, values)


def profile_metadata(
    *,
    potential_profile: str | None = None,
    potential_radius: float | None = None,
    potential_v0: float | None = None,
    potential_xi: float | None = None,
) -> dict:
    return {
        "potential_profile": normalize_potential_profile(potential_profile),
        "potential_radius": potential_radius,
        "potential_v0": potential_v0,
        "potential_xi": potential_xi,
    }
