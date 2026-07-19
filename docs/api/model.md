# Model

A {py:class}`~qbp.Model` is the object every runner drives. It bundles a real-space Hamiltonian builder, an optional momentum-space (Bloch) Hamiltonian, the optimizer / mapper / ansatz triple used by the quantum methods, and a set of observables into one registrable unit. Once a model is registered it can be driven by name through {py:func}`~qbp.run`, exactly like a built-in. For the conceptual walkthrough and the two declarative shortcuts, see [Defining a Model in Python](../models/custom-python.md), [Defining a Model in YAML](../models/custom-yaml.md), and the [programmatic builder](../models/tight-binding-builder.md).

```{eval-rst}
.. autoclass:: qbp.Model
   :members:
   :show-inheritance:
```

An {py:class}`~qbp.Observable` is a scalar (or per-band) quantity computed from a diagonalized Hamiltonian and selected with `run(observable="...")`. Every model inherits a set of built-in observables and can be given more through the constructor's `observables` argument. See [Observables](../components/observables.md) for the evaluator contract, the built-in defaults, and a worked custom observable.

```{eval-rst}
.. autoclass:: qbp.Observable
   :members:
```

## Examples

A minimal spinless model needs only its identity, lattice shape, and a real-space Hamiltonian builder:

```{code-block} python
import numpy as np
import qbp
from qbp import Model


def ssh_hamiltonian(lattice, t1, t2, boundary="periodic"):
    (n_cells,) = lattice
    H = np.zeros((2 * n_cells, 2 * n_cells), dtype=complex)
    for c in range(n_cells):
        a, b = 2 * c, 2 * c + 1
        H[a, b] += -t1
        H[b, a] += -t1
        nxt = (c + 1) % n_cells
        if nxt == 0 and boundary != "periodic":
            continue
        H[b, 2 * nxt] += -t2
        H[2 * nxt, b] += -t2
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

Add a `bloch_hamiltonian` to enable [band-structure](../more-examples/band-structure.md) runs, and pass an `observables` mapping to attach custom quantities:

```{code-block} python
from qbp._model import Observable  # also available as qbp.Observable


def staggered_density(model, lattice, H, eigvals, eigvecs, n_occ, params):
    occupied = eigvecs[:, :n_occ]
    rho = np.real(np.einsum("ij,ij->i", occupied.conj(), occupied))
    return float(rho[0::2].sum() - rho[1::2].sum())


ssh_plus = Model(
    name="ssh_obs",
    display_name="SSH (with observable)",
    param_labels={"t1": "t_1", "t2": "t_2"},
    spin=1,
    n_dims=1,
    lattice_shape=("Lx",),
    sites_per_cell=2,
    sublattices=("A", "B"),
    hamiltonian_matrix=ssh_hamiltonian,
    observables={
        "sublattice_polarization": Observable(
            name="sublattice_polarization",
            display_name=r"P_{AB}",
            analytic=staggered_density,
        ),
    },
)
```

See [Defining a Model in Python](../models/custom-python.md) for the full walkthrough, including the Bloch Hamiltonian and the optimizer / mapper / ansatz overrides.
