from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qbp._real_space import real_space_positions


@dataclass(frozen=True)
class GeometryProjection:
    geometry: str
    site_mask: np.ndarray
    orbital_mask: np.ndarray
    site_indices: np.ndarray
    positions: np.ndarray


def normalize_geometry(geometry: str | None) -> str:
    if geometry is None:
        return "rectangle"
    mode = str(geometry).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rect": "rectangle",
        "rectangular": "rectangle",
        "parallelogram": "rectangle",
        "flake": "rectangle",
        "circle": "disk",
        "circular": "disk",
        "dot": "disk",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("rectangle", "disk"):
        raise ValueError(f"unsupported geometry {geometry!r}; expected 'rectangle' or 'disk'.")
    return mode


def _disk_center(xy: np.ndarray, center) -> np.ndarray:
    if center is not None:
        arr = np.asarray(center, dtype=float)
        if arr.shape != (2,):
            raise ValueError(f"disk center must have two coordinates; got {center!r}.")
        return arr
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    return 0.5 * (lo + hi)


def geometry_projection(
    model,
    lattice,
    *,
    geometry: str | None = None,
    radius: float | None = None,
    center=None,
) -> GeometryProjection:
    mode = normalize_geometry(geometry)
    xy = real_space_positions(model, lattice)

    if mode == "rectangle":
        site_mask = np.ones(len(xy), dtype=bool)
    else:
        if radius is None:
            raise ValueError("geometry='disk' requires radius.")
        r = float(radius)
        if r <= 0.0:
            raise ValueError(f"disk radius must be positive; got {radius!r}.")
        c = _disk_center(xy, center)
        site_mask = np.linalg.norm(xy - c[None, :], axis=1) <= r
        if not np.any(site_mask):
            raise ValueError(
                f"disk geometry selected zero sites; increase radius or use a larger parent lattice."
            )

    orbital_mask = np.repeat(site_mask, model.spin)
    return GeometryProjection(
        geometry=mode,
        site_mask=site_mask,
        orbital_mask=orbital_mask,
        site_indices=np.flatnonzero(site_mask),
        positions=xy[site_mask],
    )


def apply_geometry_to_hamiltonian(H: np.ndarray, projection: GeometryProjection) -> np.ndarray:
    if projection.geometry == "rectangle":
        return H
    idx = np.flatnonzero(projection.orbital_mask)
    return H[np.ix_(idx, idx)]
