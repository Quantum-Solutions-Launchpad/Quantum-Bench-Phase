# Band Structure

A **band structure** plots a model's single-particle energies as a function of crystal momentum $\mathbf{k}$ rather than over a pair of Hamiltonian parameters. Instead of diagonalizing a finite real-space lattice, QBP evaluates the momentum-space (Bloch) Hamiltonian $H(\mathbf{k})$ at each point and returns its eigenvalues as bands. You trigger this simply by choosing momentum sweep axes—`k` for a one-dimensional model, `kx`/`ky` for a two-dimensional one—instead of parameter axes.

## The Bloch Hamiltonian Requirement

Band structure needs a Bloch Hamiltonian $H(\mathbf{k})$. For the [built-in models](../models/catalog.md), QBP derives one automatically from the term list—*unless* the model uses per-spin hopping (`spin_channels`), which breaks the simple momentum decomposition. That is why `haldane-honeycomb` supports band structure out of the box but `kane-mele-honeycomb` does not. For a [custom model](../models/custom-python.md), supply the `bloch_hamiltonian` callback (or the `bloch_hamiltonian` block in [YAML](../models/custom-yaml.md)). Requesting a momentum sweep on a model without one raises [`ModelCapabilityError`](../api/exceptions.md).

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

## A One-Dimensional Cut: the SSH Chain

The SSH chain has two sublattices per cell, so its Bloch Hamiltonian is a $2\times 2$ block with two bands. Sweeping the single momentum axis `k` across the Brillouin zone $[-\pi, \pi]$ traces both bands and the gap between them. With unequal hoppings $t_1 \neq t_2$ the chain is gapped:

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC],
    x_param="k",
    x_range=(-math.pi, math.pi, math.pi / 100),
    model_params={"t1": 1.0, "t2": 0.6},
)
```

The two curves are the SSH bands; the gap at $k = \pm\pi$ closes as $t_2 \to t_1$, marking the topological transition. Each cell here returns a list of eigenvalues (one per band) rather than a single scalar, which is what turns the sweep into a band plot.

## A Two-Dimensional Brillouin Zone: the Haldane Model

For a two-dimensional model, sweep both momentum components to map the bands over the full Brillouin zone. This reproduces `examples/run_band_structure.py` for the Haldane model, whose two bands are separated by a gap set by the mass $M$ and the flux $\phi$:

```{jupyter-execute}
result = qbp.run(
    model="haldane-honeycomb",
    method=[Method.ANALYTIC],
    x_param="kx",
    y_param="ky",
    x_range=(-math.pi, math.pi, math.pi / 40),
    y_range=(-math.pi, math.pi, math.pi / 40),
    model_params={"t1": 1.0, "t2": 0.05, "M": 0.2, "phi": math.pi / 2},
)
```

The two surfaces are the Haldane bands over the Brillouin zone; the minimum separation between them is the direct band gap. The same run drawn as a heatmap—pass `heatmap=True`—gives a top-down view of the lower band, which is often clearer for reading off where the gap is smallest:

```{jupyter-execute}
result = qbp.run(
    model="haldane-honeycomb",
    method=[Method.ANALYTIC],
    x_param="kx",
    y_param="ky",
    x_range=(-math.pi, math.pi, math.pi / 40),
    y_range=(-math.pi, math.pi, math.pi / 40),
    model_params={"t1": 1.0, "t2": 0.05, "M": 0.2, "phi": math.pi / 2},
    heatmap=True,
)
```

`Method.ANALYTIC` diagonalizes $H(\mathbf{k})$ exactly at each momentum point and returns every band, which is what makes it the reference for a band-structure run. `Method.VQE` and `Method.IQPE` also accept momentum axes—each $\mathbf{k}$ point becomes its own small ground-state problem, so they return the lowest band only—and running them against a noisy backend is a cheap way to see what a device does to a known-exact surface; [Incorporating Quantum Hardware](../user-guide/incorporating-quantum-hardware.md) does exactly that. `Method.DMRG` is the exception: it currently has no momentum-space path and QBP rejects it for these sweeps. To add a Bloch Hamiltonian to your own model and sweep its bands, see [Defining a Model in Python](../models/custom-python.md#adding-a-bloch-hamiltonian).
