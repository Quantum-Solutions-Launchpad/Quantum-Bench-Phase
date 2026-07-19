# Observables

An **observable** is a scalar (or per-band) quantity QBP extracts from a diagonalized Hamiltonian. The `observable` argument to [`qbp.run`](../api/runners.md) names which one to compute, and every value on a phase-diagram surface—analytic, VQE, IQPE, or DMRG—is that observable evaluated cell by cell. Ground-state energy `"E"` is the default, but the same machinery drives spectral gaps, kinetic and interaction energies, density fluctuations, and any custom quantity you attach to a model.

Each observable is an [`Observable`](../api/model.md) object. Every [`Model`](../api/model.md) starts with a set of built-in defaults, and you can merge in your own through the constructor's `observables` argument.

## The `Observable` Object

An `Observable` bundles a name, a display label, and one or two evaluator callbacks:

| Field | Meaning |
| --- | --- |
| `name` | Short identifier used as the lookup key and passed as `observable="..."`. |
| `display_name` | LaTeX label for plot axes, written *without* the surrounding `$` (e.g. `r"\Delta_{\mathrm{gap}}"`). |
| `analytic` | Real-space evaluator (required). |
| `analytic_bloch` | Momentum-space evaluator (optional; needed for [band-structure](../more-examples/band-structure.md) runs). |

The two evaluators receive the diagonalized single-particle Hamiltonian and return the quantity you want plotted:

- **`analytic(model, lattice, H, eigvals, eigvecs, n_occ, params) -> float`.** Called for every real-space cell. `H` is the single-particle Hamiltonian, `eigvals`/`eigvecs` are its ascending eigenvalues and eigenvectors (columns), `n_occ` is the number of filled modes, and `params` holds the resolved model parameters for that cell. The ground state fills the `n_occ` lowest modes, so most observables read out of `eigvals[:n_occ]` or the occupied block `eigvecs[:, :n_occ]`.
- **`analytic_bloch(model, k_tuple, H, eigvals, eigvecs, params) -> float | list[float]`.** Called for each momentum point on a band-structure sweep. Returning a list yields one value per band. Omitting it and requesting this observable on a momentum axis raises [`ModelCapabilityError`](../api/exceptions.md).

## Built-In Defaults

Every model inherits these six observables, so they work on the built-ins and on any custom model without extra wiring:

| `name` | Symbol | Quantity |
| --- | --- | --- |
| `E` | $E$ | Ground-state energy, $\sum_{k<N_\text{occ}} \varepsilon_k$ plus any mean-field correction. |
| `gap` | $\Delta_{\mathrm{gap}}$ | Single-particle spectral gap, $\varepsilon_{N_\text{occ}} - \varepsilon_{N_\text{occ}-1}$. |
| `kinetic_energy` | $E_{\mathrm{kin}}$ | Sum of occupied single-particle energies (the energy without the mean-field term). |
| `interaction_energy` | $E_{\mathrm{int}}$ | Mean-field interaction contribution; zero for non-interacting models. |
| `density_variance` | $\mathrm{Var}(\langle n_i \rangle)$ | Variance of the per-site occupation across the lattice. |
| `charge_gap` | $\Delta_{\mathrm{c}}$ | Second energy difference $E(n{+}1) + E(n{-}1) - 2E(n)$, reducing to the spectral gap for non-interacting models. |

List what a given model exposes—including any correlated-model magnetization observables—from the command line:

```{code-block} console
$ qbp list observables --model haldane
```

## Writing a Custom Observable

To add your own quantity, construct an [`Observable`](../api/model.md) with an `analytic` evaluator and pass it in the model's `observables` mapping. The example below defines a **double occupancy** $D = \sum_i \langle n_{i\uparrow}\rangle\langle n_{i\downarrow}\rangle$—a mean-field estimate of how often a site holds both an up and a down electron—and attaches it to a spinful dimerized chain with a staggered onsite energy $\pm M$.

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
from qbp import Model, Method, Observable


def ionic_chain(lattice, t, M, boundary="periodic"):
    (n_cells,) = lattice
    dim = 2 * n_cells * 2                       # cells x sublattices x spin
    H = np.zeros((dim, dim), dtype=complex)

    def orb(cell, sub, s):
        return ((cell * 2 + sub) * 2) + s      # spin is the fastest index

    for c in range(n_cells):
        for s in range(2):
            H[orb(c, 0, s), orb(c, 0, s)] += +M    # A sublattice
            H[orb(c, 1, s), orb(c, 1, s)] += -M    # B sublattice
            H[orb(c, 0, s), orb(c, 1, s)] += -t    # intra-cell A <-> B
            H[orb(c, 1, s), orb(c, 0, s)] += -t
            nxt = (c + 1) % n_cells
            if nxt == 0 and boundary != "periodic":
                continue
            H[orb(c, 1, s), orb(nxt, 0, s)] += -t  # inter-cell B -> A
            H[orb(nxt, 0, s), orb(c, 1, s)] += -t
    return H


def double_occupancy(model, lattice, H, eigvals, eigvecs, n_occ, params):
    if n_occ <= 0:
        return 0.0
    occupied = eigvecs[:, :n_occ]
    rho = np.real(np.einsum("ij,ij->i", occupied.conj(), occupied))
    return float(np.sum(rho[0::2] * rho[1::2]))   # up density x down density
```

The evaluator follows the contract exactly: it reads the occupied block `eigvecs[:, :n_occ]`, forms the per-orbital density $\rho_{jj} = \sum_{k<N_\text{occ}} |\langle j|\psi_k\rangle|^2$, and multiplies the even (up) entries by the odd (down) entries. Wrap it in an `Observable` and hand it to the model:

```{jupyter-execute}
model = Model(
    name="ionic_chain",
    display_name="Ionic chain",
    param_labels={"t": "t", "M": "M"},
    spin=2,
    n_dims=1,
    lattice_shape=("Lx",),
    sites_per_cell=2,
    sublattices=("A", "B"),
    hamiltonian_matrix=ionic_chain,
    observables={
        "double_occ": Observable(
            name="double_occ",
            display_name="D",
            analytic=double_occupancy,
        ),
    },
)
qbp.register_model(model)
```

Once it's registered, you can select the custom observable with `observable="double_occ"` exactly as you would a built-in. Sweeping the staggered energy $M$ shows the double occupancy rising as charge localizes onto the lower sublattice:

```{jupyter-execute}
result = qbp.run(
    model="ionic_chain",
    method=[Method.ANALYTIC],
    lattice=(4,),
    x_param="M",
    x_range=(0.0, 3.0, 0.25),
    model_params={"t": 1.0},
    observable="double_occ",
)
```

Custom observables don't replace the six defaults, so `"E"`, `"gap"`, and the rest remain available on the same model. If the model also supports [band structure](../more-examples/band-structure.md), give the `Observable` an `analytic_bloch` callback to make it available on momentum-axis sweeps too. Declarative expression-based observables are available in the [YAML schema](../models/custom-yaml.md), but you can reach for a Python `Observable` when the quantity needs real NumPy over the eigenvectors, as double occupancy does.
