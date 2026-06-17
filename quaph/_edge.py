from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quaph._realspace import (
    _normalize_boundary,
    _resolve_lattice,
    _resolve_model,
    _site_density,
    _with_boundary,
)


@dataclass
class EdgeSpectrumResult:
    model_name: str
    lattice: tuple[int, ...]
    boundary: str
    geometry: str
    edge_mask: np.ndarray
    eigvals: np.ndarray
    edge_participation: np.ndarray
    inverse_participation_ratio: np.ndarray
    plot_path: str | None = None
    figure: Any = field(default=None, repr=False)


def _site_degrees(H: np.ndarray, spin: int, *, tol: float = 1e-10) -> np.ndarray:
    n_sites = H.shape[0] // spin
    degrees = np.zeros(n_sites, dtype=int)
    for i in range(n_sites):
        a = slice(i * spin, (i + 1) * spin)
        for j in range(n_sites):
            if i == j:
                continue
            b = slice(j * spin, (j + 1) * spin)
            if (
                np.max(np.abs(H[a, b])) > tol
                or np.max(np.abs(H[b, a])) > tol
            ):
                degrees[i] += 1
    return degrees


def edge_mask_from_missing_bonds(
    H: np.ndarray,
    H_reference: np.ndarray,
    spin: int,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Mark sites that lost hopping connectivity relative to a reference matrix."""
    if H.shape != H_reference.shape:
        raise ValueError(
            f"H shape {H.shape} does not match reference shape {H_reference.shape}."
        )
    return _site_degrees(H, spin, tol=tol) < _site_degrees(H_reference, spin, tol=tol)


def projected_edge_mask_from_missing_bonds(
    H: np.ndarray,
    H_reference: np.ndarray,
    active_site_mask: np.ndarray,
    spin: int,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Mark projected sites whose degree is lower than in the unprojected reference."""
    ref_degrees = _site_degrees(H_reference, spin, tol=tol)[active_site_mask]
    return _site_degrees(H, spin, tol=tol) < ref_degrees


def edge_participation_all(eigvecs: np.ndarray, edge_mask: np.ndarray, spin: int) -> np.ndarray:
    vals = np.zeros(eigvecs.shape[1], dtype=float)
    for k in range(eigvecs.shape[1]):
        rho = _site_density(eigvecs[:, k], spin)
        vals[k] = float(rho[edge_mask].sum())
    return vals


def inverse_participation_ratio_all(eigvecs: np.ndarray, spin: int) -> np.ndarray:
    vals = np.zeros(eigvecs.shape[1], dtype=float)
    for k in range(eigvecs.shape[1]):
        rho = _site_density(eigvecs[:, k], spin)
        vals[k] = float(np.sum(rho * rho))
    return vals


def _save_and_show(fig, output_path, hide_plot):
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if not hide_plot:
        import matplotlib.pyplot as plt
        plt.show()
    return fig


def _plot_edge_spectrum(
    eigvals: np.ndarray,
    edge_participation: np.ndarray,
    ipr: np.ndarray,
    *,
    title: str,
    output_path=None,
    hide_plot: bool = False,
):
    import matplotlib.pyplot as plt

    x = np.arange(len(eigvals))
    ipr_scale = ipr / ipr.max() if ipr.size and ipr.max() > 0.0 else ipr
    sizes = 28.0 + 140.0 * ipr_scale

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    sc = ax.scatter(
        x,
        eigvals,
        c=edge_participation,
        s=sizes,
        cmap="viridis",
        edgecolors="#202020",
        linewidths=0.25,
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.75)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("edge participation")
    ax.set_xlabel("eigenstate index")
    ax.set_ylabel("energy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)


def plot_edge_spectrum(
    model,
    lattice,
    *,
    model_params: dict | None = None,
    boundary: str | None = "hard_wall",
    geometry: str | None = None,
    radius: float | None = None,
    center=None,
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
    output_path=None,
    hide_plot: bool = False,
) -> EdgeSpectrumResult:
    """Plot eigenenergies colored by boundary-site participation."""
    model = _resolve_model(model)
    lat = _resolve_lattice(model, lattice)
    boundary_mode = _normalize_boundary(boundary)
    params = _with_boundary(model_params, boundary_mode)
    boundary_mode = _normalize_boundary(params.get("boundary", boundary_mode))

    H_full = model._build_H_matrix(lat, **params)

    ref_params = dict(params)
    ref_params["boundary"] = "periodic"
    H_ref = model._build_H_matrix(lat, **ref_params)

    from quaph._geometry import apply_geometry_to_hamiltonian, geometry_projection
    projection = geometry_projection(
        model,
        lat,
        geometry=geometry,
        radius=radius,
        center=center,
    )
    H = apply_geometry_to_hamiltonian(H_full, projection)
    from quaph._profiles import apply_profiles_to_hamiltonian, normalize_mass_profile, normalize_potential_profile
    H = apply_profiles_to_hamiltonian(
        H,
        model,
        projection,
        params,
        potential_profile=potential_profile,
        potential_radius=potential_radius,
        potential_v0=potential_v0,
        potential_xi=potential_xi,
        mass_profile=mass_profile,
        mass_radius=mass_radius,
        mass_inner=mass_inner,
        mass_outer=mass_outer,
        mass_xi=mass_xi,
        profile_center=profile_center,
    )

    if projection.geometry == "rectangle":
        edge_mask = edge_mask_from_missing_bonds(H, H_ref, model.spin)
    else:
        edge_mask = projected_edge_mask_from_missing_bonds(
            H,
            H_ref,
            projection.site_mask,
            model.spin,
        )
    eigvals, eigvecs = np.linalg.eigh(H)
    edge_part = edge_participation_all(eigvecs, edge_mask, model.spin)
    ipr = inverse_participation_ratio_all(eigvecs, model.spin)

    title = (
        f"{model.display_name} edge spectrum | "
        f"{projection.geometry} | {boundary_mode.replace('_', '-')} | "
        f"V={normalize_potential_profile(potential_profile)} | "
        f"M={normalize_mass_profile(mass_profile)} | "
        f"edge sites={int(edge_mask.sum())}"
    )
    fig = _plot_edge_spectrum(
        eigvals,
        edge_part,
        ipr,
        title=title,
        output_path=output_path,
        hide_plot=hide_plot,
    )

    return EdgeSpectrumResult(
        model_name=model.name,
        lattice=lat,
        boundary=boundary_mode,
        geometry=projection.geometry,
        edge_mask=edge_mask,
        eigvals=eigvals,
        edge_participation=edge_part,
        inverse_participation_ratio=ipr,
        plot_path=str(output_path) if output_path else None,
        figure=fig,
    )
