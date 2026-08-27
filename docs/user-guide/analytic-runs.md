# Analytic Computation

Every QBP run starts from the same entry point, [`qbp.run`](../api/runners.md). Passing `method=[Method.ANALYTIC]` asks for an *analytic* computation: QBP builds the tight-binding Hamiltonian for each point in your sweep, diagonalizes it exactly, and reads the requested observable off the spectrum. There is no ansatz, no optimizer, and no sampling noise—this is the exact-diagonalization baseline that every quantum method is ultimately measured against.

Analytic runs can be cheap for small lattices, so they are the natural place to start when you're exploring a model's phase diagram or deciding which observable and parameter ranges are worth a full quantum simulation.

## Anatomy of a Run

A minimal analytic call names a model, fixes the lattice, and chooses what to sweep:

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

```{jupyter-execute}
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC],
    lattice=(8,),
    x_param="t1",
    x_range=(0.0, 2.0, 0.01),
    model_params={"t2": 1.0},
    observable="gap",
)
```

The keyword arguments break down as follows:

- **`model`.** A registered model name (`"ssh"`, `"haldane-honeycomb"`, `"hubbard-triangular"`, …) or a [`Model`](../api/model.md) instance. Built-in 2D models are registered once per lattice, as `<model>-<lattice>`; see the [model catalog](../models/catalog.md). Run `qbp list models` to see what's available.
- **`method`.** A [`Method`](../api/method.md) or list of them. For an analytic run this is `[Method.ANALYTIC]`; adding quantum methods overlays them on the same axes (see [Performing Simulation](performing-simulation.md)).
- **`lattice`.** The lattice extents, one integer per spatial dimension. The SSH chain is one-dimensional, so `lattice=(8,)` is an eight-cell chain; a two-dimensional model like Haldane takes `lattice=(3, 3)`.
- **`x_param` / `x_range`.** The horizontal sweep axis and its `(min, max, step)` range. Any model parameter, `n_occ` (filling), or a momentum axis is fair game.
- **`y_param` / `y_range`.** An optional second sweep axis. Provide it to turn a 1D curve into a 2D surface; omit it for a line plot.
- **`model_params`.** Fixed values for every parameter that isn't being swept. A parameter cannot appear both here and as a sweep axis—the active axis owns it.
- **`observable`.** Which quantity to read off each spectrum. Defaults to the ground-state energy `"E"`.
- **`log_path` / `plot_path`.** Exact JSON and PDF files to write; either may be omitted. See [Logging and Reloading](logging-and-reloading.md).
- **`heatmap`.** Render a 2D sweep as a flat heatmap instead of a 3D surface.

The call returns a [`RunResult`](../api/results.md) that holds the computed grid and knows how to plot itself. When `hide_plot` is left `False`, the figure is drawn immediately, which is why the block above renders without any explicit `.plot()` call. [Results and Plotting](results-and-plotting.md) covers the object in detail.

## Sweeping Over Filling

The `n_occ` axis is special: it sweeps the number of occupied single-particle levels rather than a Hamiltonian parameter, so it needs no `x_range`. Left unbounded, it walks from the empty lattice up to full occupation. Here is the ground-state energy of an SSH chain as a function of filling:

```{jupyter-execute}
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC],
    lattice=(8,),
    x_param="n_occ",
    model_params={"t1": 1.0, "t2": 0.4},
)
```

The energy falls as electrons fill the lower band, bottoms out near half-filling, and climbs back up as the upper band fills—the familiar signature of a two-band model. When `n_occ` is not being swept, QBP defaults to half-filling.

## Two-Dimensional Sweeps

Adding a `y_param` produces a surface. Sweeping filling against the inter-cell hopping $t_2$ shows how the energy landscape deforms as the chain crosses its topological transition at $t_1 = t_2$:

```{jupyter-execute}
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC],
    lattice=(4,),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 2.0, 0.25),
    model_params={"t1": 1.0},
)
```

For a two-parameter scan a heatmap is often easier to read than a surface. Passing `heatmap=True` flattens the same data into a color map; it requires both sweep axes and exactly one method:

```{jupyter-execute}
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC],
    lattice=(8,),
    x_param="t1",
    x_range=(0.0, 2.0, 0.02),
    y_param="t2",
    y_range=(0.0, 2.0, 0.02),
    observable="gap",
    heatmap=True,
)
```

The spectral gap collapses to zero along the diagonal $t_1 = t_2$, tracing the phase boundary as a dark valley.

## Choosing an Observable

By default QBP computes the ground-state energy `"E"`. Every built-in model also exposes a handful of other observables—the spectral gap `"gap"`, the charge gap `"charge_gap"`, the kinetic and interaction energies `"kinetic_energy"` / `"interaction_energy"`, and the density variance `"density_variance"`—and correlated models add magnetization observables on top. List what a given model supports with:

```{code-block} console
$ qbp list observables --model haldane-honeycomb
```
