# Defining a Model in Python

When the [built-in catalog](catalog.md) doesn't cover what you need, you can define your own [`Model`](../api/model.md) directly in Python. A `Model` bundles everything the runners need into one object: a real-space Hamiltonian builder, an optional momentum-space (Bloch) Hamiltonian, an optimizer / mapper / ansatz triple for the quantum methods, and a set of observables. Once registered, a custom model is a first-class citizen—every [`qbp.run`](../api/runners.md) feature (analytic sweeps, VQE/IQPE, heatmaps, logging) works exactly as it does for the built-ins.

For purely declarative tight-binding models, the [YAML](custom-yaml.md) and [programmatic builder](tight-binding-builder.md) routes are shorter. Reach for the full Python constructor when you need a Hamiltonian that isn't a sum of standard hopping/onsite terms, or when you want to attach custom observables or factories.

## The Constructor

At minimum a model needs its identity, its lattice shape, and a `hamiltonian_matrix` builder:

| Argument | Meaning |
| --- | --- |
| `name` | Unique registry key, e.g. `"ssh"`. |
| `display_name` | Human-readable label for plots and the console. |
| `param_labels` | Map of parameter name → LaTeX label (without `$`). |
| `spin` | `1` (spinless) or `2` (spinful). |
| `n_dims` | Spatial dimensionality: `1`, `2`, or `3`. |
| `lattice_shape` | Names of the lattice extents, one per dimension. |
| `sites_per_cell` | Number of sublattice sites per unit cell. |
| `hamiltonian_matrix` | Real-space Hamiltonian builder (required). |

Everything else is optional and falls back to sensible defaults—a Jordan–Wigner mapper, an SPSA optimizer, an excitation-preserving ansatz, and the standard observables (`E`, `gap`, `kinetic_energy`, `interaction_energy`, `density_variance`, `charge_gap`). The most useful optional arguments are `sublattices` (names for the per-cell sites), `bloch_hamiltonian` (for band structure), `interaction_hamiltonian` and `mean_field_correction` (for interacting models), `get_optimizer` / `get_mapper` / `get_vqe_ansatz` (to override the quantum-method stack), and `observables` (to attach custom quantities). See the [`Model` API reference](../api/model.md) for the full signature.

The two Hamiltonian callbacks are the heart of the object:

- **`hamiltonian_matrix(lattice, **params)`** returns the single-particle Hamiltonian as a dense `(N, N)` array in the real-space site basis, where `N` counts spin-orbitals. It is **required**. QBP threads the resolved boundary condition into `params` as a `boundary` key (`"periodic"` or `"open"`), so your builder should accept it—either as an explicit keyword or by absorbing `**params`. Real-space runs, open boundaries, and all quantum methods flow through this callback.
- **`bloch_hamiltonian(*ks, **params)`** returns the `(sites_per_cell, sites_per_cell)` Bloch Hamiltonian $H(\mathbf{k})$ at crystal momentum `ks`. It is **optional**; provide it only if you want momentum-space [band structure](../more-examples/band-structure.md) runs. Requesting a band structure on a model without it raises [`ModelCapabilityError`](../api/exceptions.md).

Provide **`hamiltonian_matrix`** alone for a finite-lattice model, add **`bloch_hamiltonian`** when you also want band structures, and there is no reason to provide the Bloch Hamiltonian without the real-space one—the latter is what the runners diagonalize for every non-band-structure sweep.

## Worked Example: the SSH Chain

The following defines the SSH chain from scratch, following `examples/run_custom_model.py`. Each unit cell holds two sublattice sites, `A` and `B`; the intra-cell bond carries $-t_1$ and the inter-cell bond carries $-t_2$.

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

```{jupyter-execute}
import numpy as np
import qbp
from qbp import Model, Method


def ssh_hamiltonian(lattice, t1, t2, boundary="periodic"):
    (n_cells,) = lattice
    H = np.zeros((2 * n_cells, 2 * n_cells), dtype=complex)
    for c in range(n_cells):
        a, b = 2 * c, 2 * c + 1
        H[a, b] += -t1            # intra-cell A <-> B
        H[b, a] += -t1
        nxt = c + 1
        if nxt == n_cells:
            if boundary != "periodic":
                continue          # open chain: no wrap-around bond
            nxt = 0
        a2 = 2 * nxt
        H[b, a2] += -t2           # inter-cell B -> A of next cell
        H[a2, b] += -t2
    return H


ssh = Model(
    name="ssh_custom",
    display_name="SSH (custom)",
    param_labels={"t1": "t_1", "t2": "t_2"},
    spin=1,
    n_dims=1,
    lattice_shape=("Lx",),
    sites_per_cell=2,
    sublattices=("A", "B"),
    hamiltonian_matrix=ssh_hamiltonian,
)

qbp.register_model(ssh)
```

Registering the model adds it to the runtime registry under its `name`, so it can now be driven by name just like a built-in:

```{jupyter-execute}
result = qbp.run(
    model="ssh_custom",
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

## Adding a Bloch Hamiltonian

To support band-structure runs, supply `bloch_hamiltonian`. For the SSH chain the Bloch Hamiltonian is the familiar $2\times 2$ block

$$
H(k) = \begin{pmatrix} 0 & f(k) \\ f^*(k) & 0 \end{pmatrix},
\qquad f(k) = -t_1 - t_2 e^{ik}.
$$

```{jupyter-execute}
import math


def ssh_bloch(k, t1, t2, boundary="periodic"):
    f = -t1 - t2 * np.exp(1j * k)
    return np.array([[0.0, f], [np.conj(f), 0.0]], dtype=complex)


ssh_bs = Model(
    name="ssh_bs",
    display_name="SSH (band structure)",
    param_labels={"t1": "t_1", "t2": "t_2"},
    spin=1,
    n_dims=1,
    lattice_shape=("Lx",),
    sites_per_cell=2,
    sublattices=("A", "B"),
    hamiltonian_matrix=ssh_hamiltonian,
    bloch_hamiltonian=ssh_bloch,
)
qbp.register_model(ssh_bs)

result = qbp.run(
    model="ssh_bs",
    method=[Method.ANALYTIC],
    x_param="k",
    x_range=(-math.pi, math.pi, math.pi / 100),
    model_params={"t1": 1.0, "t2": 0.6},
)
```

Sweeping the momentum axis `k` traces the two SSH bands and the gap between them. To attach custom observables (for example a double-occupancy or magnetization), pass an `observables` mapping into the constructor; see [Observables](../components/observables.md) for the evaluator contract and the defaults every model inherits.
