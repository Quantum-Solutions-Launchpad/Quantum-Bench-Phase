# Boundary Conditions with the Haldane Model

QBP supports real-space boundary-condition studies in the same `run()` workflow
used for phase-diagram sweeps. This page shows the common Haldane-model cases:
periodic bulk runs, open hard-wall flakes, disk-shaped quantum dots, soft
radial confinement, and a topological interface imposed through a radial
Semenoff-mass profile.

The public selector is deliberately small:

| Setting | Meaning | Extra parameters |
|---|---|---|
| `boundary="periodic"` | Wrap lattice edges and preserve translation symmetry. | None |
| `boundary="open"` | Build a finite real-space Hamiltonian without wraparound links. | Optional `boundary_params` |

`boundary_params` are only valid with `boundary="open"`. The accepted keys are
`geometry`, `radius`, `center`, `potential_profile`, `potential_radius`,
`potential_v0`, and `potential_xi`.

```{note}
Open boundaries are real-space features. Do not combine them with momentum
sweep axes such as `kx` or `ky`; band-structure runs require periodic
boundaries. For open-boundary spatial diagnostics, use `x_param="Lx"` and
`y_param="Ly"` for a state-density plot, or `x_param="eigenstate"` for an
edge-participation spectrum.
```

The command-line examples below use decimal approximations for common angles:
`pi / 4 = 0.7853981634` and `pi / 2 = 1.5707963268`.

```{jupyter-execute}
:hide-code:
:hide-output:
:stderr:

import os
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp())

import matplotlib.pyplot as plt
from loguru import logger
from IPython.display import Image, display

logger.remove()

def show_run(result):
    """Display one QBP figure explicitly for jupyter-sphinx."""
    fig = result.plot(hide_plot=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image_path = tmp.name
    fig.savefig(image_path, bbox_inches="tight", dpi=160)
    display(Image(filename=image_path))
    plt.close(fig)
    return result
```

## Hard-Wall Open Flake

The simplest open-boundary run keeps the full rectangular parent lattice but
omits any hopping that would wrap through the boundary. This is the direct
open-boundary analogue of a periodic real-space sweep.

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(3, 3),
    boundary="open",
    x_param="n_occ",
    x_range=(0, 6, 1),
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 4, "M": 0.0},
    hide_plot=True,
))
```

The same run from the command line:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 3 3 \
  --boundary open \
  --x-param n_occ \
  --x-range 0 6 1 \
  --t1 1.0 \
  --t2 0.1 \
  --phi 0.7853981634 \
  --M 0.0 \
  --log-path examples/logs/haldane/3x3/hard-wall-n_occ.json \
  --plot-path examples/plots/haldane/3x3/hard-wall-n_occ.pdf \
  --hide-plot
```

Use this form when you want the whole finite flake and only need to switch
between periodic and open connectivity.

## Disk Hard-Wall Dot

To carve a finite dot from a larger parent flake, keep `boundary="open"` and
set `boundary_params={"geometry": "disk", "radius": ...}`. The disk projection
keeps sites inside the chosen radius and drops the rest of the Hamiltonian. The
edge of the retained disk is therefore a hard wall.

This cell renders the real-space density of the state closest to zero energy:

```{jupyter-execute}
import math
import qbp
from qbp import Method

density = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(8, 8),
    boundary="open",
    boundary_params={"geometry": "disk", "radius": 3.2},
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 2, "M": 0.2},
    hide_plot=True,
))
```

This separate cell renders the edge-participation spectrum for the same dot:

```{jupyter-execute}
import math
import qbp
from qbp import Method

edge_spectrum = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(8, 8),
    boundary="open",
    boundary_params={"geometry": "disk", "radius": 3.2},
    x_param="eigenstate",
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 2, "M": 0.2},
    hide_plot=True,
))
```

The matching density CLI command is:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 8 8 \
  --boundary open \
  --geometry disk \
  --radius 3.2 \
  --x-param Lx \
  --y-param Ly \
  --t1 1.0 \
  --t2 0.1 \
  --phi 1.5707963268 \
  --M 0.2 \
  --plot-path examples/plots/haldane/disk/disk-density-hard-wall.pdf \
  --hide-plot
```

The matching edge-spectrum CLI command is:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 8 8 \
  --boundary open \
  --geometry disk \
  --radius 3.2 \
  --x-param eigenstate \
  --t1 1.0 \
  --t2 0.1 \
  --phi 1.5707963268 \
  --M 0.2 \
  --plot-path examples/plots/haldane/disk/disk-edge-spectrum-hard-wall.pdf \
  --hide-plot
```

The `Lx`/`Ly` diagnostic renders the real-space density of a single-particle
state. If you do not pass `n_occ`, QBP chooses the eigenstate closest to zero
energy; pass `n_occ` or `--n-occ` to plot the highest occupied state for a
particular filling. The `eigenstate` diagnostic plots all eigenenergies colored
by boundary participation, which is useful for identifying edge-localized
in-gap states.

## Soft Radial Confinement

A soft-dot profile adds a smooth onsite scalar potential

$$
V(r) = \frac{V_0}{2}\left[1 + \tanh\left(\frac{r - R_\text{dot}}{\xi}\right)\right],
$$

where `potential_radius` is $R_\text{dot}$, `potential_v0` is $V_0$, and
`potential_xi` controls the wall width. This creates an internal confinement
boundary inside the finite open flake.

The density diagnostic shows where the near-zero state localizes relative to
the soft confinement wall:

```{jupyter-execute}
import math
import qbp
from qbp import Method

soft_density = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(8, 8),
    boundary="open",
    boundary_params={
        "potential_profile": "soft_dot",
        "potential_radius": 3.0,
        "potential_v0": 3.0,
        "potential_xi": 0.8,
    },
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 2, "M": 0.2},
    hide_plot=True,
))
```

The edge spectrum is a separate plot because it is a different observable:

```{jupyter-execute}
import math
import qbp
from qbp import Method

soft_edge_spectrum = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(8, 8),
    boundary="open",
    boundary_params={
        "potential_profile": "soft_dot",
        "potential_radius": 3.0,
        "potential_v0": 3.0,
        "potential_xi": 0.8,
    },
    x_param="eigenstate",
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 2, "M": 0.2},
    hide_plot=True,
))
```

The matching density CLI command is:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 8 8 \
  --boundary open \
  --potential-profile soft-dot \
  --potential-radius 3.0 \
  --potential-v0 3.0 \
  --potential-xi 0.8 \
  --x-param Lx \
  --y-param Ly \
  --t1 1.0 \
  --t2 0.1 \
  --phi 1.5707963268 \
  --M 0.2 \
  --plot-path examples/plots/haldane/8x8/soft-dot-density.pdf \
  --hide-plot
```

The matching edge-spectrum CLI command is:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 8 8 \
  --boundary open \
  --potential-profile soft-dot \
  --potential-radius 3.0 \
  --potential-v0 3.0 \
  --potential-xi 0.8 \
  --x-param eigenstate \
  --t1 1.0 \
  --t2 0.1 \
  --phi 1.5707963268 \
  --M 0.2 \
  --plot-path examples/plots/haldane/8x8/soft-dot-edge-spectrum.pdf \
  --hide-plot
```

Use the soft profile when the physics should distinguish a physical outer edge
from a smoother confinement wall inside the sample.

## Radial Semenoff-Mass Interface

The soft-dot potential is model-independent: it adds a scalar onsite wall. The
Haldane model also supports a model-specific topological interface through
`SemenoffMass`, which replaces the uniform staggered mass with a radial profile
$M(r)$. A sharp `radial_step` or smooth `radial_tanh` profile separates an inner
mass from an outer mass at the selected radius.

Set the base model parameter `M` to `0.0` when the radial profile supplies the
full mass. This cell renders the density plot:

```{jupyter-execute}
import math
import qbp
from qbp import Method, SemenoffMass

interface_density = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(8, 8),
    boundary="open",
    x_param="Lx",
    y_param="Ly",
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 2, "M": 0.0},
    investigation=SemenoffMass(
        profile="radial_tanh",
        radius=3.0,
        inner=0.2,
        outer=0.8,
        xi=0.8,
    ),
    hide_plot=True,
))
```

This separate cell renders the edge-participation spectrum:

```{jupyter-execute}
import math
import qbp
from qbp import Method, SemenoffMass

interface_edge_spectrum = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(8, 8),
    boundary="open",
    x_param="eigenstate",
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 2, "M": 0.0},
    investigation=SemenoffMass(
        profile="radial_tanh",
        radius=3.0,
        inner=0.2,
        outer=0.8,
        xi=0.8,
    ),
    hide_plot=True,
))
```

The density CLI selects the same investigation by name:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 8 8 \
  --boundary open \
  --x-param Lx \
  --y-param Ly \
  --investigation semenoff_mass \
  --semenoff-mass-profile radial_tanh \
  --semenoff-mass-radius 3.0 \
  --semenoff-mass-inner 0.2 \
  --semenoff-mass-outer 0.8 \
  --semenoff-mass-xi 0.8 \
  --t1 1.0 \
  --t2 0.1 \
  --phi 1.5707963268 \
  --M 0.0 \
  --plot-path examples/plots/haldane/8x8/topological-interface-density.pdf \
  --hide-plot
```

The edge-spectrum CLI uses the same investigation parameters with
`x_param="eigenstate"`:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic \
  --lattice 8 8 \
  --boundary open \
  --x-param eigenstate \
  --investigation semenoff_mass \
  --semenoff-mass-profile radial_tanh \
  --semenoff-mass-radius 3.0 \
  --semenoff-mass-inner 0.2 \
  --semenoff-mass-outer 0.8 \
  --semenoff-mass-xi 0.8 \
  --t1 1.0 \
  --t2 0.1 \
  --phi 1.5707963268 \
  --M 0.0 \
  --plot-path examples/plots/haldane/8x8/topological-interface-edge-spectrum.pdf \
  --hide-plot
```

Use this investigation when the boundary is not merely geometric, but separates
two Haldane parameter regimes.

## Open Boundaries with Quantum Methods

Boundary conditions also flow into the real-space fermionic Hamiltonian used by
the quantum-method pipeline. Keep the lattice small for VQE/IQPE examples,
because every retained spin-orbital becomes a qubit after mapping.

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = show_run(qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE],
    lattice=(1, 2),
    boundary="open",
    x_param="n_occ",
    x_range=(1, 4, 1),
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 4, "M": 0.0},
    method_params={
        Method.VQE: {"iters": 20, "layers": 1, "reps": 1},
    },
    hide_plot=True,
))
```

CLI equivalent:

```{code-block} shell
qbp run \
  --model haldane \
  --method analytic vqe \
  --lattice 1 2 \
  --boundary open \
  --x-param n_occ \
  --x-range 1 4 1 \
  --t1 1.0 \
  --t2 0.1 \
  --phi 0.7853981634 \
  --M 0.0 \
  --vqe-iters 20 \
  --vqe-layers 1 \
  --vqe-reps 1 \
  --hide-plot
```

This is the same API shape as a periodic VQE run. Only the boundary selector
changes the Hamiltonian QBP maps to qubits.

## Common Checks

- `boundary_params` with `boundary="periodic"` is an error. Periodic boundaries
  take no parameters.
- `geometry="disk"` requires `radius`; `center` is optional and defaults to the
  center of the parent lattice.
- `potential_profile="soft_dot"` requires `potential_radius` and `potential_v0`;
  `potential_xi` defaults to `0.8` if omitted.
- The `Lx`/`Ly` and `eigenstate` diagnostics are analytic single-particle plots,
  so run them with `method=Method.ANALYTIC` or `--method analytic`.
- Open-boundary, disk, soft-dot, and investigation settings are for real-space
  lattice runs. They are rejected for `kx`/`ky` band-structure sweeps.

For the lower-level helper functions behind these examples, see
[Real-Space and Boundary Analysis](../api/spatial-analysis.md). For the
`SemenoffMass` investigation contract, see [Investigations](../api/investigations.md).
