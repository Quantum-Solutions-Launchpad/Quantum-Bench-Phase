# Real-Space and Boundary Analysis

Under open boundary conditions, QBP works directly with the real-space site basis. These helpers support that regime: they place sites in the plane, carve finite quantum-dot geometries out of a flake, add smooth confinement potentials, and measure how strongly each eigenstate localizes on the boundary. Most users reach these effects through {py:func}`~qbp.run`'s `boundary` and `boundary_params` arguments; the functions here are the lower-level building blocks, useful when you want to assemble and diagnose an open-boundary Hamiltonian yourself. See the [open-boundary concepts](../getting-started/concepts.md) page for more information.

## Site Coordinates

```{eval-rst}
.. autofunction:: qbp.real_space_positions
```

```{eval-rst}
.. autoclass:: qbp.RealSpaceStateResult
   :members:
```

{py:func}`~qbp.real_space_positions` returns one `xy` coordinate per site (using the model's `lattice_vectors`/`sublattice_positions` when available), the geometric basis every other tool here builds on:

```{code-block} python
import qbp

model = qbp.get_model("haldane-honeycomb")
xy = qbp.real_space_positions(model, (6, 6))   # (n_sites, 2) coordinates
```

## Geometries

A {py:class}`~qbp.GeometryProjection` selects which sites survive—the whole flake (`rectangle`) or a disk of a given radius—and {py:func}`~qbp.apply_geometry_to_hamiltonian` restricts a Hamiltonian to them, producing the finite hard-wall dot described in the [concepts](../getting-started/concepts.md#open-boundary-conditions-and-quantum-dot-geometries).

```{eval-rst}
.. autofunction:: qbp.geometry_projection
```

```{eval-rst}
.. autoclass:: qbp.GeometryProjection
   :members:
```

```{eval-rst}
.. autofunction:: qbp.apply_geometry_to_hamiltonian
```

```{code-block} python
H = model._build_H_matrix((10, 10), t1=1.0, t2=0.1, phi=0.5, M=0.0, boundary="open")

projection = qbp.geometry_projection(model, (10, 10), geometry="disk", radius=4.0)
H_dot = qbp.apply_geometry_to_hamiltonian(H, projection)   # Hamiltonian of the retained sites
```

## Confinement Potentials

{py:func}`~qbp.soft_dot_potential` evaluates the smooth radial barrier $V(r) = \tfrac{v_0}{2}[1 + \tanh((r - R)/\xi)]$ per site, and {py:func}`~qbp.apply_profiles_to_hamiltonian` adds it to a Hamiltonian's diagonal to define a soft-confinement dot.

```{eval-rst}
.. autofunction:: qbp.soft_dot_potential
```

```{eval-rst}
.. autofunction:: qbp.apply_profiles_to_hamiltonian
```

```{code-block} python
H_soft = qbp.apply_profiles_to_hamiltonian(
    H_dot, model, projection, {"boundary": "open"},
    potential_profile="soft_dot", potential_radius=3.0, potential_v0=5.0, potential_xi=0.8,
)
```

## Edge-State Metrics

Once you have an open-boundary Hamiltonian, these three functions quantify boundary localization. {py:func}`~qbp.edge_mask_from_missing_bonds` flags the boundary sites (those that lost bonds relative to a periodic reference), {py:func}`~qbp.edge_participation_all` measures how much of each eigenstate sits on them, and {py:func}`~qbp.inverse_participation_ratio_all` measures how localized each eigenstate is overall. The {py:class}`~qbp.EdgeSpectrumResult` bundles these into the edge-participation spectrum.

```{eval-rst}
.. autofunction:: qbp.edge_mask_from_missing_bonds
```

```{eval-rst}
.. autofunction:: qbp.edge_participation_all
```

```{eval-rst}
.. autofunction:: qbp.inverse_participation_ratio_all
```

```{eval-rst}
.. autoclass:: qbp.EdgeSpectrumResult
   :members:
```

```{code-block} python
import numpy as np

H_ref = model._build_H_matrix((10, 10), t1=1.0, t2=0.1, phi=0.5, M=0.0, boundary="periodic")
edge_mask = qbp.edge_mask_from_missing_bonds(H_dot, H_ref, model.spin)

eigvals, eigvecs = np.linalg.eigh(H_dot)
edge_part = qbp.edge_participation_all(eigvecs, edge_mask, model.spin)
ipr = qbp.inverse_participation_ratio_all(eigvecs, model.spin)
```

States with both high edge participation and high IPR are the boundary-localized modes—the chiral edge states of a topological phase.
