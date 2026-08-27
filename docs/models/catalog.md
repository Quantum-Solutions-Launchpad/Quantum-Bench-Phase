# Built-In Models

QBP currently has six built-in tight-binding models, registered across twelve lattice variants. Every 2D model ships on more than one lattice, and each lattice configuration is its own YAML file under `qbp/models/` and its own registry entry, named `<model>-<lattice>`; the 1D SSH chain is the one model with a single lattice, so it keeps the bare name `ssh`. Pass that name as the `model` argument to [`qbp.run`](../api/runners.md) and you are ready to sweep phase diagrams and benchmark quantum methods. Run `qbp list models` to see what is registered in your environment.

The built-in specs are also worked examples: copy one into your own YAML file and edit it to define a variant—a new lattice for an existing model is often a handful of changed offsets (see [Defining a Model in YAML](custom-yaml.md))—build the equivalent structure programmatically with [`build_tight_binding_model`](tight-binding-builder.md), or drop to the full Python API for arbitrary Hamiltonians (see [Defining a Model in Python](custom-python.md)). The per-model pages below give each model's Hamiltonian, its lattices, its parameters, its canonical sweep, and a runnable snippet.

## Overview

| Model | Lattices | Dim | Spin | Interacting | Phase Diagram | Band Structure |
| --- | --- | --- | --- | --- | --- | --- |
| [SSH](ssh.md)                         | `ssh`                                                                | 1D | spinless | no  | yes | yes |
| [Haldane](haldane.md)                 | `haldane-honeycomb`, `haldane-square`                                | 2D | spinless | no  | yes | yes |
| [Kane–Mele](kane-mele.md)             | `kane-mele-honeycomb`, `kane-mele-square`                            | 2D | spinful  | no  | yes | —   |
| [Kane–Mele (Liu–Chen)](kane-mele-lc.md) | `kane-mele-lc-honeycomb`, `kane-mele-lc-square`                    | 2D | spinful  | no  | yes | —   |
| [Hubbard](hubbard.md)                 | `hubbard-honeycomb`, `hubbard-square`, `hubbard-triangular`          | 2D | spinful  | yes | yes | —   |
| [Haldane–Hubbard](haldane-hubbard.md) | `haldane-hubbard-honeycomb`, `haldane-hubbard-square`                | 2D | spinful  | yes | yes | —   |

The two Kane–Mele families use per-spin hopping (`spin_channels`), which blocks the automatic momentum-space decomposition, so they have no band-structure path; see [Band Structure](../more-examples/band-structure.md).

## Lattices

A lattice variant changes only the geometry block of the YAML—`lattice_vectors`, `sublattice_positions`, and the hopping `offsets`. The parameters, the observables, and the sweep API are the same across a model's lattices, so switching lattice is a one-word change:

```python
qbp.run(model="hubbard-honeycomb", ...)    # z = 3, bipartite
qbp.run(model="hubbard-square", ...)       # z = 4, bipartite
qbp.run(model="hubbard-triangular", ...)   # z = 6, frustrated
```

| Lattice | Cell | Coordination | Bipartite | Notes |
| --- | --- | --- | --- | --- |
| honeycomb   | 2 sites ($A$, $B$) | 3 | yes | The original built-in geometry; Dirac points at $K$, $K'$. |
| square      | 2 sites ($A$, $B$) | 4 | yes | $A$/$B$ are the two checkerboard sublattices, so the staggered mass and Néel order carry over unchanged. |
| triangular  | 1 site             | 6 | no  | Frustrated: no sublattice splitting, no Néel order, spectrum bounded by $[-6t, 3t]$. |

`lattice=(Lx, Ly)` always counts **unit cells**, not sites, so a $2\times2$ honeycomb or square lattice is 8 sites while a $2\times2$ triangular lattice is 4 sites.

Not every model exists on every lattice. The Haldane and Kane–Mele families are built on a staggered sublattice mass $\pm M$ and on a chirality $\nu_{ij}$ defined by triangular loops of bonds; both need a two-site cell, so those models ship on honeycomb and square only. The triangular lattice has a one-site cell and no sublattice to stagger, so it is offered where frustration is the point of the model—the Hubbard model.

```{toctree}
:hidden:
:maxdepth: 1

ssh
haldane
kane-mele
kane-mele-lc
hubbard
haldane-hubbard
```
