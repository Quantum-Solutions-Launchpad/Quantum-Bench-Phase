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
    r0 = float(radius)
    width = max(float(xi), 1e-12)
    r = _radial_distances(xy, center)
    return 0.5 * float(v0) * (1.0 + np.tanh((r - r0) / width))


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


def _add_site_diagonal(H: np.ndarray, spin: int, site_values: np.ndarray) -> np.ndarray:
    out = H.copy()
    for site, value in enumerate(site_values):
        for s in range(spin):
            orbital = site * spin + s
            out[orbital, orbital] += value
    return out


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
    mass_profile: str | None = None,
    mass_radius: float | None = None,
    mass_inner: float | None = None,
    mass_outer: float | None = None,
    mass_xi: float | None = None,
    profile_center=None,
) -> np.ndarray:
    out = H
    potential_mode = normalize_potential_profile(potential_profile)
    if potential_mode != "none":
        if potential_radius is None or potential_v0 is None:
            raise ValueError("soft-dot potential requires potential_radius and potential_v0.")
        values = soft_dot_potential(
            projection.positions,
            radius=potential_radius,
            v0=potential_v0,
            xi=0.8 if potential_xi is None else potential_xi,
            center=profile_center,
        )
        out = _add_site_diagonal(out, model.spin, values)

    mass_mode = normalize_mass_profile(mass_profile)
    if mass_mode != "none":
        if mass_radius is None or mass_inner is None or mass_outer is None:
            raise ValueError("radial mass profile requires mass_radius, mass_inner, and mass_outer.")
        if "M" not in model_params:
            raise ValueError("radial mass profile requires the base model parameter 'M'.")
        target_mass = radial_mass_values(
            projection.positions,
            profile=mass_mode,
            radius=mass_radius,
            mass_inner=mass_inner,
            mass_outer=mass_outer,
            xi=mass_xi,
            center=profile_center,
        )
        base_mass = float(model_params["M"])
        delta_mass = target_mass - base_mass
        out = _add_site_diagonal(out, model.spin, delta_mass * _sublattice_signs(model, projection))

    return out


def profile_metadata(
    *,
    potential_profile: str | None = None,
    potential_radius: float | None = None,
    potential_v0: float | None = None,
    potential_xi: float | None = None,
    mass_profile: str | None = None,
    mass_radius: float | None = None,
    mass_inner: float | None = None,
    mass_outer: float | None = None,
    mass_xi: float | None = None,
    profile_center=None,
) -> dict:
    return {
        "potential_profile": normalize_potential_profile(potential_profile),
        "potential_radius": potential_radius,
        "potential_v0": potential_v0,
        "potential_xi": potential_xi,
        "mass_profile": normalize_mass_profile(mass_profile),
        "mass_radius": mass_radius,
        "mass_inner": mass_inner,
        "mass_outer": mass_outer,
        "mass_xi": mass_xi,
        "profile_center": list(profile_center) if profile_center is not None else None,
    }
