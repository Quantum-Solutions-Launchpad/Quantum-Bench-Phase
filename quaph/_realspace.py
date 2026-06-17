from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quaph._model import Model
from quaph._registry import get_model as _get_model


@dataclass
class RealSpaceStateResult:
    model_name: str
    lattice: tuple[int, ...]
    boundary: str
    geometry: str
    state_index: int
    energy: float
    positions: np.ndarray
    densities: np.ndarray
    eigvals: np.ndarray
    plot_path: str | None = None
    figure: Any = field(default=None, repr=False)


def _resolve_model(model) -> Model:
    if isinstance(model, str):
        return _get_model(model)
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance or model name string, got {type(model).__name__}")
    return model


def _resolve_lattice(model: Model, lattice) -> tuple[int, ...]:
    try:
        lat = tuple(int(x) for x in lattice)
    except TypeError:
        raise ValueError(
            f"lattice must be a sequence of ints matching {model.lattice_shape}; got {lattice!r}"
        )
    if len(lat) != model.n_dims:
        raise ValueError(
            f"Model '{model.name}' expects lattice with {model.n_dims} entries "
            f"(shape={model.lattice_shape}); got lattice={lat}."
        )
    if any(x < 1 for x in lat):
        raise ValueError(f"lattice entries must be >= 1; got {lat}.")
    return lat


def _normalize_boundary(boundary: str | None) -> str:
    if boundary is None:
        return "periodic"
    mode = str(boundary).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pbc": "periodic",
        "periodic_boundary": "periodic",
        "periodic_boundary_condition": "periodic",
        "open": "hard_wall",
        "obc": "hard_wall",
        "hardwall": "hard_wall",
        "hard_wall_boundary": "hard_wall",
        "hard_wall_boundary_condition": "hard_wall",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("periodic", "hard_wall"):
        raise ValueError(
            f"unsupported boundary mode {boundary!r}; expected 'periodic' or 'hard_wall'."
        )
    return mode


def _with_boundary(params: dict | None, boundary: str | None) -> dict:
    out = dict(params or {})
    requested = _normalize_boundary(boundary)
    existing_key = "boundary" if "boundary" in out else "boundary_condition" if "boundary_condition" in out else None
    if existing_key is not None:
        existing = _normalize_boundary(out[existing_key])
        if requested != "periodic" and existing != requested:
            raise ValueError(
                f"conflicting boundary settings: boundary={requested!r}, "
                f"model_params[{existing_key!r}]={existing!r}."
            )
        out.pop("boundary_condition", None)
        out["boundary"] = existing
    elif requested != "periodic":
        out["boundary"] = requested
    return out


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


def real_space_positions(model: Model, lattice) -> np.ndarray:
    """Return one xy coordinate per real-space site, with spin channels collapsed."""
    lat = _resolve_lattice(model, lattice)
    n_sites = int(np.prod(lat)) * model.sites_per_cell
    xy = np.zeros((n_sites, 2), dtype=float)

    use_cartesian = model.lattice_vectors is not None
    if use_cartesian:
        a_mat = np.array(model.lattice_vectors, dtype=float)
        sub_pos = {
            name: np.array(model.sublattice_positions.get(name, (0.0,) * model.n_dims), dtype=float)
            for name in model.sublattices
        }

    for cell in _iter_cells(lat):
        cell_flat = _flat_cell_index(cell, lat)
        for sub_idx, sub_name in enumerate(model.sublattices):
            site_idx = cell_flat * model.sites_per_cell + sub_idx
            if use_cartesian:
                pos = np.array(cell, dtype=float) @ a_mat + sub_pos[sub_name]
                if pos.size == 1:
                    xy[site_idx] = (pos[0], 0.0)
                else:
                    xy[site_idx] = pos[:2]
            else:
                base = np.array(cell[:2] if len(cell) >= 2 else (cell[0], 0.0), dtype=float)
                offset = np.array([(sub_idx % 2) * 0.35, (sub_idx // 2) * 0.35], dtype=float)
                xy[site_idx] = base + offset
    return xy


def _site_density(eigvec: np.ndarray, spin: int) -> np.ndarray:
    if eigvec.size % spin != 0:
        raise ValueError(
            f"eigenvector length {eigvec.size} is not divisible by model spin={spin}."
        )
    rho = np.abs(eigvec.reshape((-1, spin))) ** 2
    density = np.real(rho.sum(axis=1))
    total = float(density.sum())
    return density / total if total > 0.0 else density


def _extract_bonds(H: np.ndarray, spin: int, *, tol: float = 1e-10, max_bonds: int = 3000):
    n_sites = H.shape[0] // spin
    bonds: list[tuple[int, int, float]] = []
    for i in range(n_sites):
        a = slice(i * spin, (i + 1) * spin)
        for j in range(i + 1, n_sites):
            b = slice(j * spin, (j + 1) * spin)
            strength = max(float(np.max(np.abs(H[a, b]))), float(np.max(np.abs(H[b, a]))))
            if strength > tol:
                bonds.append((i, j, strength))
                if len(bonds) >= max_bonds:
                    return bonds
    return bonds


def _choose_state(eigvals: np.ndarray, state_index: int | None, n_occ: int | None) -> int:
    if state_index is not None and n_occ is not None:
        raise ValueError("Use either state_index or n_occ, not both.")
    if state_index is not None:
        k = int(state_index)
    elif n_occ is not None:
        if n_occ <= 0:
            raise ValueError("n_occ must be positive when selecting the highest occupied state.")
        k = int(n_occ) - 1
    else:
        k = int(np.argmin(np.abs(eigvals)))
    if k < 0 or k >= len(eigvals):
        raise ValueError(f"state_index={k} is outside the valid range [0, {len(eigvals) - 1}].")
    return k


def _save_and_show(fig, output_path, hide_plot):
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if not hide_plot:
        import matplotlib.pyplot as plt
        plt.show()
    return fig


def _plot_density_2d(
    xy: np.ndarray,
    density: np.ndarray,
    bonds,
    *,
    title: str,
    output_path=None,
    hide_plot: bool = False,
    show_bonds: bool = True,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    max_strength = max((b[2] for b in bonds), default=1.0)
    if show_bonds:
        for i, j, strength in bonds:
            lw = 0.35 + 0.8 * strength / max_strength
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#b7b7b7", linewidth=lw, alpha=0.55)

    vmax = float(density.max()) if density.size else 1.0
    sizes = 45.0 + 520.0 * density / vmax if vmax > 0.0 else np.full_like(density, 45.0)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=density, s=sizes, cmap="viridis", edgecolors="#202020", linewidths=0.35)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(r"$|\psi_i|^2$")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)


def _plot_density_3d(
    xy: np.ndarray,
    density: np.ndarray,
    bonds,
    *,
    title: str,
    output_path=None,
    hide_plot: bool = False,
    show_bonds: bool = True,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    max_strength = max((b[2] for b in bonds), default=1.0)
    if show_bonds:
        for i, j, strength in bonds:
            lw = 0.35 + 0.9 * strength / max_strength
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                [0.0, 0.0],
                color="#b9b9b9",
                linewidth=lw,
                alpha=0.45,
            )

    vmax = float(density.max()) if density.size else 1.0
    for (x, y), z in zip(xy, density):
        ax.plot([x, x], [y, y], [0.0, z], color="#444444", linewidth=0.55, alpha=0.55)

    sizes = 35.0 + 580.0 * density / vmax if vmax > 0.0 else np.full_like(density, 35.0)
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], density,
        c=density,
        s=sizes,
        cmap="viridis",
        edgecolors="#202020",
        linewidths=0.25,
        depthshade=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.08, fraction=0.04)
    cbar.set_label(r"$|\psi_i|^2$")

    ax.set_title(title, pad=18)
    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel(r"$|\psi_i|^2$", labelpad=8)
    ax.set_zlim(0.0, max(vmax * 1.15, 1e-12))
    ax.view_init(elev=28, azim=-55)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return _save_and_show(fig, output_path, hide_plot)


def plot_real_space_state_density(
    model,
    lattice,
    *,
    model_params: dict | None = None,
    boundary: str | None = None,
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
    state_index: int | None = None,
    n_occ: int | None = None,
    view: str = "2d",
    show_bonds: bool = True,
    max_bonds: int = 3000,
    output_path=None,
    hide_plot: bool = False,
) -> RealSpaceStateResult:
    """Plot a real-space single-particle eigenstate density for a YAML model."""
    model = _resolve_model(model)
    lat = _resolve_lattice(model, lattice)
    boundary_mode = _normalize_boundary(boundary)
    params = _with_boundary(model_params, boundary_mode)
    boundary_mode = _normalize_boundary(params.get("boundary", boundary_mode))

    H = model._build_H_matrix(lat, **params)
    from quaph._geometry import apply_geometry_to_hamiltonian, geometry_projection
    projection = geometry_projection(
        model,
        lat,
        geometry=geometry,
        radius=radius,
        center=center,
    )
    H = apply_geometry_to_hamiltonian(H, projection)
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
    eigvals, eigvecs = np.linalg.eigh(H)
    k = _choose_state(eigvals, state_index, n_occ)

    xy = projection.positions
    density = _site_density(eigvecs[:, k], model.spin)
    bonds = _extract_bonds(H, model.spin, max_bonds=max_bonds)

    title = (
        f"{model.display_name} real-space eigenstate density | "
        f"{projection.geometry} | {boundary_mode.replace('_', '-')} | "
        f"V={normalize_potential_profile(potential_profile)} | "
        f"M={normalize_mass_profile(mass_profile)} | "
        f"state={k} | E={eigvals[k]:+.6f}"
    )
    view_mode = str(view).strip().lower()
    if view_mode in ("3d", "three_d", "surface"):
        fig = _plot_density_3d(
            xy, density, bonds, title=title, output_path=output_path,
            hide_plot=hide_plot, show_bonds=show_bonds,
        )
    elif view_mode in ("2d", "planar", "flat"):
        fig = _plot_density_2d(
            xy, density, bonds, title=title, output_path=output_path,
            hide_plot=hide_plot, show_bonds=show_bonds,
        )
    else:
        raise ValueError("view must be '3d' or '2d'.")

    return RealSpaceStateResult(
        model_name=model.name,
        lattice=lat,
        boundary=boundary_mode,
        geometry=projection.geometry,
        state_index=k,
        energy=float(eigvals[k]),
        positions=xy,
        densities=density,
        eigvals=eigvals,
        plot_path=str(output_path) if output_path else None,
        figure=fig,
    )
