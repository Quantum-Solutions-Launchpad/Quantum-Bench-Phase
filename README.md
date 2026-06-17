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

## Open Boundary Conditions

Real-space YAML models support an optional boundary mode for finite lattice
runs. `boundary` selects `"periodic"` (default, matching the original behavior)
or `"open"`, which removes hopping terms that would wrap around the finite
unit-cell grid. These are the only two accepted values.

The open-boundary domain shape and potential profile are configured through a
`boundary_params` dict, paired with `boundary` the same way `model_params` is
paired with `model`. Periodic boundaries take no parameters, so `boundary_params`
must be omitted (or empty) for them.

Python API:

```python
import qbp
from qbp import Method

qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(3, 3),
    boundary="open",
    x_param="n_occ",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
)
```

CLI:

```bash
qbp run --model haldane --method analytic --lattice 3 3 --boundary open \
  --x-param n_occ --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0
```

This is the Phase 1 open-boundary implementation: the Hilbert-space size is
unchanged, but boundary-crossing hopping and bond interaction terms are omitted
instead of wrapped periodically.

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
    boundary="open",
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
    plot_path="examples/plots/haldane/3x3/real-space-density-hard-wall-2d.pdf",
    hide_plot=True,
)
```

CLI:

```bash
qbp run --model haldane --method analytic --lattice 3 3 --boundary open \
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
localized on an open boundary. The diagnostic builds the requested real-space
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
    boundary="open",
    x_param="eigenstate",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.2},
    plot_path="examples/plots/haldane/6x6/edge-spectrum-hard-wall.pdf",
    hide_plot=True,
)
```

CLI:

```bash
qbp run --model haldane --method analytic --lattice 6 6 --boundary open \
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
lattice bounding box. In the Python API `geometry`/`radius`/`center` live inside
`boundary_params` (an open-boundary domain shape):

```python
qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(14, 14),
    boundary="open",
    boundary_params={"geometry": "disk", "radius": 5.5},
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.2},
    plot_path="examples/plots/haldane/disk/disk-density-hard-wall.pdf",
    hide_plot=True,
)
```

```bash
qbp run --model haldane --method analytic --lattice 14 14 --boundary open \
  --geometry disk --radius 5.5 --x-param Lx --y-param Ly \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2 \
  --plot-path examples/plots/haldane/disk/disk-density-hard-wall.pdf --hide-plot
```

The same geometry works for edge participation (`--x-param eigenstate`) and for
analytic energy sweeps (`--x-param n_occ`). This disk implementation projects the
parent tight-binding Hamiltonian onto the active disk sites. It is currently
intended for single-particle analytic and diagnostic workflows; simulated/VQE
disk support should be added later by building a projected many-body operator.

## Open-Boundary Soft Confinement

Real-space analytic and diagnostic workflows support an open-boundary radial
confinement potential `V(r)`, a single-particle tight-binding diagnostic
matching the Haldane-dot Phase-2 picture. It adds a scalar onsite potential to
every active site, so the `potential_*` knobs live inside `boundary_params`:

```text
V(r) = 0.5 * V0 * [1 + tanh((r - R) / xi)]
```

```python
qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(18, 18),
    boundary="open",
    boundary_params={
        "potential_profile": "soft_dot",
        "potential_radius": 5.5,
        "potential_v0": 3.0,
        "potential_xi": 0.8,
    },
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.2},
)
```

```bash
qbp run --model haldane --method analytic --lattice 18 18 --boundary open \
  --x-param Lx --y-param Ly \
  --potential-profile soft-dot --potential-radius 5.5 \
  --potential-v0 3.0 --potential-xi 0.8 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2
```

## Investigations: Model-Specific Physics Studies

Beyond the generic boundary and potential knobs, a run can apply an
*investigation* — a model-specific modification of the Hamiltonian, the physics
analogue of how `method` selects a solver. Investigations are pluggable: each
lives in its own module, declares its parameters, gates itself on model
capability, and registers itself, so adding a study needs no change to
`qbp.run`. Select one by instance or by registered name, with parameters carried
in `investigation_params` (mirroring `method` / `method_params`):

```python
from qbp import SemenoffMass

qbp.run(
    model="haldane",
    method=Method.ANALYTIC,
    lattice=(18, 18),
    boundary="open",
    investigation=SemenoffMass(profile="radial_tanh", radius=5.5,
                               inner=0.2, outer=0.8, xi=0.8),
    x_param="eigenstate",
    # M = 0.0 so the radial profile supplies the full mass.
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.0},
)

# Equivalent, selecting by registered name:
qbp.run(
    model="haldane", method=Method.ANALYTIC, lattice=(18, 18), boundary="open",
    investigation="semenoff_mass",
    investigation_params={"profile": "radial_tanh", "radius": 5.5,
                          "inner": 0.2, "outer": 0.8, "xi": 0.8},
    x_param="eigenstate",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.0},
)
```

The only investigation bundled today is `semenoff_mass`: a radial Semenoff-mass
interface `M(r)` for Haldane-like A/B lattices. It adds `M(r) - M` to the A/B
onsite mass terms, so the base model parameter `M` should usually be `0.0` when
the profile supplies the full mass. Applying it to a model without A/B
sublattices raises a `ModelCapabilityError`.

In the CLI, `--investigation <name>` selects the study and each investigation's
parameters become `--<name>-<param>` flags:

```bash
qbp run --model haldane --method analytic --lattice 18 18 --boundary open \
  --x-param eigenstate \
  --investigation semenoff_mass \
  --semenoff-mass-profile radial_tanh --semenoff-mass-radius 5.5 \
  --semenoff-mass-inner 0.2 --semenoff-mass-outer 0.8 --semenoff-mass-xi 0.8 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.0
```

Open-boundary potentials and investigations can be combined with
`--geometry disk` when the desired domain is a circular open-boundary dot rather
than a rectangular/parallelogram flake.

## Parameter Reference

All parameters below are keyword arguments of `qbp.run(...)`; each has a matching
`--kebab-case` CLI flag (e.g. `potential_v0` → `--potential-v0`).

Boundary:

- `boundary` — finite-lattice boundary condition. Exactly two values are
  accepted: `"periodic"` (default) wraps hopping/interaction terms around the
  grid; `"open"` drops the wrap-around terms, leaving real edges.
- `boundary_params` — dict of open-boundary settings, paired with `boundary` the
  same way `model_params` is paired with `model`. Valid only when
  `boundary="open"`; periodic boundaries take no parameters. Accepted keys are
  the domain and potential settings below. In the CLI these are individual flags
  (`--geometry`, `--radius`, ...) collected into `boundary_params` automatically.

Open-boundary domain (`boundary_params` keys):

- `geometry` — active-site domain shape inside the parent `lattice`.
  `"rectangle"` (default) keeps the full parallelogram flake; `"disk"` keeps only
  sites within `radius` of `center`.
- `radius` — disk radius in real-space coordinates. Required when
  `geometry="disk"`; ignored otherwise.
- `center` — `(x, y)` disk center in real-space coordinates. Defaults to the
  center of the parent lattice bounding box.

Open-boundary scalar potential profile `V(r)` (`boundary_params` keys; added to
the onsite energy of active sites):

- `potential_profile` — `"none"` (default) or `"soft_dot"`. Selects whether a
  radial confinement potential is added.
- `potential_radius` — wall radius `R` in `V(r) = 0.5 * V0 * [1 + tanh((r - R)/xi)]`.
- `potential_v0` — outer potential height `V0` (well depth/barrier).
- `potential_xi` — smoothing length `xi` of the wall (smaller is sharper).
- The potential is centered on the domain `center` above (the disk center, or
  the parent lattice center when unset).

Investigation (model-specific study):

- `investigation` — an `Investigation` instance or registered name (currently
  `"semenoff_mass"`), applied to real-space analytic runs and diagnostics. Pairs
  with `investigation_params` the way `method` pairs with `method_params`.
- `investigation_params` — parameter dict, used when selecting by name.

The `semenoff_mass` investigation (radial Semenoff mass `M(r)` for Haldane-like
A/B lattices) takes:

- `profile` — `"radial_step"` (sharp interface) or `"radial_tanh"` (default,
  smoothed interface).
- `radius` — interface radius separating the inner and outer mass regions.
- `inner` — mass value inside the interface.
- `outer` — mass value outside the interface.
- `xi` — smoothing length, used by `"radial_tanh"`.
- `center` — `(x, y)` profile center; defaults to the active geometry center.
- It adds `M(r) - M` to the base mass, so set the model parameter `M = 0.0` when
  the profile should supply the full mass.

Eigenstate selection (real-space density only):

- `n_occ` — plot the highest occupied state for filling `N`, i.e. eigenstate
  index `N - 1`. When omitted, the state closest to `E = 0` is shown.
