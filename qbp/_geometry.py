from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qbp._real_space import real_space_positions


@dataclass(frozen=True)
class GeometryProjection:
    """A selection of lattice sites carving a finite geometry out of a flake.

    Produced by :func:`geometry_projection`. ``site_mask`` / ``orbital_mask``
    flag the retained sites and spin-orbitals of the parent lattice,
    ``site_indices`` lists the kept site indices, and ``positions`` gives the
    Cartesian coordinates of the retained sites. Pass it to
    :func:`apply_geometry_to_hamiltonian` to restrict a Hamiltonian.
    """

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
    """Select the sites of ``model`` on ``lattice`` that fall inside a geometry.

    ``geometry`` is ``'rectangle'`` (keep the whole flake, the default) or
    ``'disk'`` (keep sites within ``radius`` of ``center``, defaulting to the
    flake center). Returns a :class:`GeometryProjection` describing the retained
    sites, which :func:`apply_geometry_to_hamiltonian` uses to build the finite
    open-boundary Hamiltonian.
    """
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
    """Restrict a single-particle Hamiltonian to a geometry's retained orbitals.

    Returns the submatrix of ``H`` on the spin-orbitals kept by ``projection``.
    A ``'rectangle'`` projection keeps everything and returns ``H`` unchanged; a
    ``'disk'`` projection returns the smaller Hamiltonian of the selected sites.
    """
    if projection.geometry == "rectangle":
        return H
    idx = np.flatnonzero(projection.orbital_mask)
    return H[np.ix_(idx, idx)]


def orbital_index_map(projection: GeometryProjection) -> np.ndarray:
    mask = projection.orbital_mask
    mapping = np.full(mask.shape[0], -1, dtype=int)
    mapping[np.flatnonzero(mask)] = np.arange(int(mask.sum()))
    return mapping


def project_fermionic_op(op, projection: GeometryProjection | None):
    if projection is None or projection.geometry == "rectangle":
        return op
    from qiskit_nature.second_q.operators import FermionicOp

    mapping = orbital_index_map(projection)
    n_active = int(projection.orbital_mask.sum())
    new_terms: dict[str, complex] = {}
    for label, coeff in op.items():
        if not label:
            new_terms[label] = new_terms.get(label, 0j) + coeff
            continue
        tokens = label.split()
        translated = []
        drop = False
        for token in tokens:
            action, idx = token.split("_")
            j = mapping[int(idx)]
            if j < 0:
                drop = True
                break
            translated.append(f"{action}_{j}")
        if drop:
            continue
        new_label = " ".join(translated)
        new_terms[new_label] = new_terms.get(new_label, 0j) + coeff
    return FermionicOp(new_terms, num_spin_orbitals=n_active)
