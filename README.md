# QSL-NNL-P7
QSL-NNL Project 7 : Quantum Benchmarking of Majorana Systems

## Hard-Wall Boundary Conditions

Real-space YAML models support an optional boundary mode for finite lattice
runs. The default is periodic, matching the original behavior. Use
`hard_wall`/`hard-wall`/`open` to remove hopping terms that would wrap around
the finite unit-cell grid.

Python API:

```python
qbp.run_analytic(
    model="haldane",
    lattice=(3, 3),
    boundary="hard_wall",
    x_param="n_occ",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
)
```

CLI:

```bash
qbp run analytic --model haldane --lattice 3 3 --boundary hard-wall \
  --x-param n_occ --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0
```

This is the Phase 1 hard-wall/open-boundary implementation: the Hilbert-space
size is unchanged, but boundary-crossing hopping and bond interaction terms are
omitted instead of wrapped periodically.

## Real-Space Eigenstate Density Plots

Use `plot-state` to visualize a single-particle eigenstate density on the
finite real-space lattice. The default view is a 2D real-space plot: site
position is drawn in the model's `x/y` coordinates, and the normalized density
`|psi_i|^2` is shown with a colorbar.

```bash
qbp plot-state --model haldane --lattice 3 3 --boundary hard-wall \
  --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0
```

Save the plot instead of opening a window:

```bash
qbp plot-state --model haldane --lattice 3 3 --boundary hard-wall \
  --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0 \
  --output examples/plots/haldane/3x3/real-space-density-hard-wall-2d.pdf \
  --hide-plot
```

Python API:

```python
qbp.plot_real_space_state_density(
    model="haldane",
    lattice=(3, 3),
    boundary="hard_wall",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
)
```

By default the state closest to zero energy is selected. Use `--state-index K`
for an exact eigenstate, or `--n-occ N` to plot the highest occupied state
`N - 1`. For interacting YAML models, this plot is a single-particle diagnostic
of the tight-binding matrix; interaction terms remain part of the energy and
simulation pipeline. Add `--view 3d` when a vertical density-height view is
useful.

## Edge-Spectrum Diagnostics

Use `edge-spectrum` to identify which single-particle eigenstates are localized
on a hard-wall boundary. The diagnostic builds the requested real-space
Hamiltonian, compares its site connectivity against the periodic version, marks
sites that lost bonds as edge sites, and colors each eigenenergy by

```text
edge participation = sum |psi_i|^2 over edge sites.
```

CLI:

```bash
qbp edge-spectrum --model haldane --lattice 6 6 --boundary hard-wall \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2 \
  --output examples/plots/haldane/6x6/edge-spectrum-hard-wall.pdf \
  --hide-plot
```

Python API:

```python
qbp.plot_edge_spectrum(
    model="haldane",
    lattice=(6, 6),
    boundary="hard_wall",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 1.5708, "M": 0.2},
    output_path="examples/plots/haldane/6x6/edge-spectrum-hard-wall.pdf",
    hide_plot=True,
)
```

This is still a single-particle tight-binding diagnostic. For interacting/VQE
states, a future many-body version should plot site occupations such as
`<n_i>` from the prepared ground state.

## Disk Geometry

Analytic and real-space diagnostic workflows can also use a disk-shaped domain
inside a larger parent lattice. The parent lattice is still specified with
`--lattice`; the disk mask selects active sites by real-space distance from a
center point. If no center is provided, QBP uses the center of the parent
lattice bounding box.

```bash
qbp plot-state --model haldane --lattice 14 14 --boundary hard-wall \
  --geometry disk --radius 5.5 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2 \
  --output examples/plots/haldane/disk/disk-density-hard-wall.pdf \
  --hide-plot
```

The same geometry can be used for edge participation:

```bash
qbp edge-spectrum --model haldane --lattice 14 14 --boundary hard-wall \
  --geometry disk --radius 5.5 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2 \
  --output examples/plots/haldane/disk/disk-edge-spectrum-hard-wall.pdf \
  --hide-plot
```

And for analytic sweeps:

```bash
qbp run analytic --model haldane --lattice 14 14 --boundary hard-wall \
  --geometry disk --radius 5.5 \
  --x-param n_occ --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2
```

This disk implementation projects the parent tight-binding Hamiltonian onto
the active disk sites. It is currently intended for single-particle analytic
and diagnostic workflows. Simulated/VQE disk support should be added later by
building a projected many-body operator.

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

CLI:

```bash
qbp plot-state --model haldane --lattice 18 18 --boundary hard-wall \
  --potential-profile soft-dot --potential-radius 5.5 \
  --potential-v0 3.0 --potential-xi 0.8 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.2
```

The radial mass profile changes the Haldane/Semenoff mass spatially. It is
implemented by adding the difference `M(r) - M` to the A/B onsite mass terms,
so the base model parameter `M` should usually be set to `0.0` when the radial
profile supplies the full mass:

```bash
qbp edge-spectrum --model haldane --lattice 18 18 --boundary hard-wall \
  --mass-profile radial-tanh --mass-radius 5.5 \
  --mass-inner 0.2 --mass-outer 0.8 --mass-xi 0.8 \
  --t1 1.0 --t2 0.1 --phi 1.5708 --M 0.0
```

These profile options can be combined with `--geometry disk` when the desired
domain is a circular hard-wall dot rather than a rectangular/parallelogram
flake.
