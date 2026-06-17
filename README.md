# QSL-NNL-P7
QSL-NNL Project 7 : Quantum Benchmarking of Majorana Systems

All boundary-condition workflows go through the two top-level commands: `qbp.run(...)`
in Python (`qbp run ...` on the CLI) to compute and render, and `RunResult.plot(...)`
(`qbp plot ...`) to re-render a result. The kind of figure is selected by the sweep
axes, mirroring how a momentum axis (`kx`/`ky`) selects a band-structure run:

| `x_param` / `y_param`        | Figure                                            |
| ---------------------------- | ------------------------------------------------- |
| a parameter or `n_occ`       | energy sweep (the `observable` over the sweep)    |
| `kx` / `ky`                  | band structure                                    |
| `Lx` / `Ly` (lattice axes)   | real-space single-particle eigenstate density     |
| `eigenstate`                 | edge-participation spectrum                        |

The real-space diagnostics (`Lx`/`Ly` and `eigenstate`) are single-particle
exact-diagonalization plots of one Hamiltonian, so they require
`method=Method.ANALYTIC` and take no sweep range. The boundary condition,
geometry, and radial profiles described below all apply to them.

## Hard-Wall Boundary Conditions

Real-space YAML models support an optional boundary mode for finite lattice
runs. The default is periodic, matching the original behavior. Use
`hard_wall`/`hard-wall`/`open` to remove hopping terms that would wrap around
the finite unit-cell grid.

Python API:

```python
import qbp
from qbp import Method

qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(3, 3),
    boundary="hard_wall",
    x_param="n_occ",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
)
```

CLI:

```bash
qbp run --model haldane --method analytic --lattice 3 3 --boundary hard-wall \
  --x-param n_occ --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0
```

This is the Phase 1 hard-wall/open-boundary implementation: the Hilbert-space
size is unchanged, but boundary-crossing hopping and bond interaction terms are
omitted instead of wrapped periodically.

## Real-Space Eigenstate Density Plots

Sweep the real-space lattice axes (`Lx`/`Ly`) to visualize a single-particle
eigenstate density on the finite lattice. Site position is drawn in the model's
`x/y` coordinates, and the normalized density `|psi_i|^2` is shown with a
colorbar.

Python API:

```python
qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(3, 3),
    boundary="hard_wall",
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
    plot_path="examples/plots/haldane/3x3/real-space-density-hard-wall-2d.pdf",
    hide_plot=True,
)
```

CLI:

```bash
qbp run --model haldane --method analytic --lattice 3 3 --boundary hard-wall \
  --x-param Lx --y-param Ly --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0 \
  --plot-path examples/plots/haldane/3x3/real-space-density-hard-wall-2d.pdf --hide-plot
```

By default the state closest to zero energy is selected. Use `n_occ=N`
(`--n-occ N`) to plot the highest occupied state `N - 1` instead. For interacting
YAML models, this plot is a single-particle diagnostic of the tight-binding
matrix; interaction terms remain part of the energy and simulation pipeline. The
call returns a `RunResult`; `result.diagnostic` holds the `RealSpaceStateResult`
(positions, densities, eigenvalues), and `result.plot(output_path=..., hide_plot=True)`
re-renders it.

## Edge-Spectrum Diagnostics

Sweep the `eigenstate` axis to identify which single-particle eigenstates are
localized on a hard-wall boundary. The diagnostic builds the requested real-space
Hamiltonian, compares its site connectivity against the periodic version, marks
sites that lost bonds as edge sites, and colors each eigenenergy by

```text
edge participation = sum |psi_i|^2 over edge sites.
```

Python API:

```python
qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(6, 6),
    boundary="hard_wall",
    x_param="eigenstate",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.2},
    plot_path="examples/plots/haldane/6x6/edge-spectrum-hard-wall.pdf",
    hide_plot=True,
)
```

CLI:

```bash
qbp run --model haldane --method analytic --lattice 6 6 --boundary hard-wall \
  --x-param eigenstate --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2 \
  --plot-path examples/plots/haldane/6x6/edge-spectrum-hard-wall.pdf --hide-plot
```

This is still a single-particle tight-binding diagnostic. For interacting/VQE
states, a future many-body version should plot site occupations such as
`<n_i>` from the prepared ground state.

## Disk Geometry

Real-space runs (energy sweeps and both diagnostics) can also use a disk-shaped
domain inside a larger parent lattice. The parent lattice is still specified with
`lattice`; the disk mask selects active sites by real-space distance from a
center point. If no center is provided, QBP uses the center of the parent
lattice bounding box.

```bash
qbp run --model haldane --method analytic --lattice 14 14 --boundary hard-wall \
  --geometry disk --radius 5.5 --x-param Lx --y-param Ly \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2 \
  --plot-path examples/plots/haldane/disk/disk-density-hard-wall.pdf --hide-plot
```

The same geometry works for edge participation (`--x-param eigenstate`) and for
analytic energy sweeps (`--x-param n_occ`). This disk implementation projects the
parent tight-binding Hamiltonian onto the active disk sites. It is currently
intended for single-particle analytic and diagnostic workflows; simulated/VQE
disk support should be added later by building a projected many-body operator.

## Radial Profiles: Soft Confinement and Topological Interfaces

Real-space analytic and diagnostic workflows also support radial onsite
profiles. These are single-particle tight-binding diagnostics intended to match
the Haldane-dot phases:

```text
Phase 2: scalar soft confinement V(r)
Phase 3: radial Semenoff mass profile M(r)
```

Soft confinement adds a scalar onsite potential to every active site:

```text
V(r) = 0.5 * V0 * [1 + tanh((r - R) / xi)]
```

```bash
qbp run --model haldane --method analytic --lattice 18 18 --boundary hard-wall \
  --x-param Lx --y-param Ly \
  --potential-profile soft-dot --potential-radius 5.5 \
  --potential-v0 3.0 --potential-xi 0.8 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2
```

The radial mass profile changes the Haldane/Semenoff mass spatially. It is
implemented by adding the difference `M(r) - M` to the A/B onsite mass terms,
so the base model parameter `M` should usually be set to `0.0` when the radial
profile supplies the full mass:

```bash
qbp run --model haldane --method analytic --lattice 18 18 --boundary hard-wall \
  --x-param eigenstate \
  --mass-profile radial-tanh --mass-radius 5.5 \
  --mass-inner 0.2 --mass-outer 0.8 --mass-xi 0.8 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.0
```

These profile options can be combined with `--geometry disk` when the desired
domain is a circular hard-wall dot rather than a rectangular/parallelogram
flake.

## Parameter Reference

All parameters below are keyword arguments of `qbp.run(...)`; each has a matching
`--kebab-case` CLI flag (e.g. `potential_v0` → `--potential-v0`).

Boundary and domain:

- `boundary` — finite-lattice boundary condition. `"periodic"` (default) wraps
  hopping/interaction terms around the grid; `"hard_wall"` (aliases `"open"`,
  `"obc"`, `"hard-wall"`) drops the wrap-around terms, leaving real edges.
- `geometry` — active-site domain shape inside the parent `lattice`.
  `"rectangle"` (default) keeps the full parallelogram flake; `"disk"` (aliases
  `"circle"`, `"dot"`) keeps only sites within `radius` of `center`.
- `radius` — disk radius in real-space coordinates. Required when
  `geometry="disk"`; ignored otherwise.
- `center` — `(x, y)` disk center in real-space coordinates. Defaults to the
  center of the parent lattice bounding box.

Scalar potential profile `V(r)` (added to the onsite energy of active sites):

- `potential_profile` — `"none"` (default) or `"soft_dot"` (aliases `"soft"`,
  `"soft_confinement"`). Selects whether a radial confinement potential is added.
- `potential_radius` — wall radius `R` in `V(r) = 0.5 * V0 * [1 + tanh((r - R)/xi)]`.
- `potential_v0` — outer potential height `V0` (well depth/barrier).
- `potential_xi` — smoothing length `xi` of the wall (smaller is sharper).

Radial Semenoff mass profile `M(r)` (Haldane-like A/B sublattice mass):

- `mass_profile` — `"none"` (default), `"radial_step"` (alias `"topological"`,
  a sharp interface), or `"radial_tanh"` (alias `"smooth"`, a smoothed interface).
- `mass_radius` — interface radius separating the inner and outer mass regions.
- `mass_inner` — mass value inside the interface.
- `mass_outer` — mass value outside the interface.
- `mass_xi` — smoothing length, used by `"radial_tanh"`.
- The profile adds `M(r) - M` to the base mass, so set the model parameter
  `M = 0.0` when the profile should supply the full mass.

- `profile_center` — `(x, y)` center for the radial potential/mass profiles.
  Defaults to the active geometry center (the disk center, or the lattice center
  for a rectangle).

Eigenstate selection (real-space density only):

- `n_occ` — plot the highest occupied state for filling `N`, i.e. eigenstate
  index `N - 1`. When omitted, the state closest to `E = 0` is shown.
