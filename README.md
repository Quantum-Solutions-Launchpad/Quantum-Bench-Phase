# QSL-NNL-P7
QSL-NNL Project 7 : Quantum Benchmarking of Majorana Systems

## Hard-Wall Boundary Conditions

Real-space YAML models support an optional boundary mode for finite lattice
runs. The default is periodic, matching the original behavior. Use
`hard_wall`/`hard-wall`/`open` to remove hopping terms that would wrap around
the finite unit-cell grid.

Python API:

```python
quaph.run_analytic(
    model="haldane",
    lattice=(3, 3),
    boundary="hard_wall",
    x_param="n_occ",
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.785, "M": 0.0},
)
```

CLI:

```bash
quaph run analytic --model haldane --lattice 3 3 --boundary hard-wall \
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
quaph plot-state --model haldane --lattice 3 3 --boundary hard-wall \
  --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0
```

Save the plot instead of opening a window:

```bash
quaph plot-state --model haldane --lattice 3 3 --boundary hard-wall \
  --t1 1.0 --t2 0.1 --phi 0.785 --M 0.0 \
  --output examples/plots/haldane/3x3/real-space-density-hard-wall-2d.png \
  --hide-plot
```

Python API:

```python
quaph.plot_real_space_state_density(
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
