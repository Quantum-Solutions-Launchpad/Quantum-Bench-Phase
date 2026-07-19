# Defining a Model in YAML

Most tight-binding models are a sum of standard hopping, onsite, and interaction terms, and for those the fastest route is a declarative YAML file. All six [built-in models](catalog.md) are defined this way under `qbp/models/`, and you can add your own by writing the same schema and registering it:

```{code-block} python
import qbp
model = qbp.register_model_from_file("my_model.yaml")
```

`register_model_from_file` validates the spec, registers the model under its `name`, and copies the file into `qbp/models/` so it persists across sessions. From then on the model is available by name to [`qbp.run`](../api/runners.md). This page is the schema reference; for a Python-first alternative that takes the same fields as function arguments, see the [programmatic builder](tight-binding-builder.md).

## Top-Level Keys

| Key | Required? | Meaning |
| --- | --- | --- |
| `name` | ✓ | Registry key, e.g. `ssh`. |
| `display_name` | ✓ | Human-readable label for plots and the console. |
| `spin` | ✓ | `1` (spinless) or `2` (spinful). |
| `n_dims` | ✓ | Spatial dimensionality: `1`, `2`, or `3`. |
| `lattice_shape` | ✓ | Names of the lattice extents, one per dimension, e.g. `[Lx, Ly]`. |
| `sites_per_cell` | ✓ | Number of sublattice sites per unit cell (≥ 1). |
| `sublattices` | ✓ | Sublattice names, length must equal `sites_per_cell`, e.g. `[A, B]`. |
| `parameters` | ✓ | Map of parameter name → `{label: "<LaTeX>"}`. |
| `terms` | ✓ | List of Hamiltonian terms (see below). |
| `optimizer` | ✗ | VQE classical optimizer: `{type, kwargs}`. |
| `mapper` | ✗ | Fermion-to-qubit mapper: `{type, kwargs}`. |
| `ansatz` | ✗ | VQE ansatz: `{type, kwargs, initial_state_prefix}`. |
| `iqpe_initial_state` | ✗ | IQPE initial-state recipe. |
| `bloch_hamiltonian` | ✗ | Explicit momentum-space Hamiltonian (see below). |
| `lattice_vectors` | ✗ | Real-space lattice vectors, one per dimension. |
| `sublattice_positions` | ✗ | Cartesian offset of each sublattice within the cell (requires `lattice_vectors`). |
| `observables` | ✗ | Extra expression-based observables. |

There is no explicit `interaction` or `mean_field_correction` key: the many-body interaction operator and its mean-field energy correction are derived automatically from any `density_density` terms you include.

## Parameters

Each entry under `parameters` names a runtime knob and gives its LaTeX label (used on plot axes, without the surrounding `$`):

```{code-block} yaml
parameters:
  t1: {label: "t_1"}
  t2: {label: "t_2"}
```

Parameter names are in scope inside every term's `coefficient` expression.

## Terms

A term's `kind` selects one of three shapes. Every `coefficient` is a string expression evaluated with NumPy in scope and the model parameters bound as names—so `-t1`, `-t2 * exp(1j*phi)`, and `+1j*lambda_SO` are all valid.

### `hopping`

A hopping term connects a `from` sublattice to a `to` sublattice across one or more cell `offsets` (integer displacements in cell coordinates, one component per dimension).

| Field | Required? | Meaning |
| --- | --- | --- |
| `from` | ✓ | Source sublattice name. |
| `to` | ✓ | Destination sublattice name. |
| `offsets` | ✓ | List of integer offset vectors, e.g. `[[0], [1]]` or `[[0, 0], [-1, 0]]`. |
| `coefficient` | ✓ | Hopping amplitude expression. |
| `hermitian_partner` | ✗ | If `true`, also add the conjugate reverse hop, so you only write one direction. Default `false`. |
| `spin_channels` | ✗ | Restrict the term to `[up]`, `[down]`, or both (spinful models only). |

```{code-block} yaml
- kind: hopping
  from: A
  to: B
  offsets: [[0]]
  coefficient: "-t1"
  hermitian_partner: true
```

### `onsite`

An onsite term adds a diagonal energy on a single sublattice—used for staggered masses and potentials.

| Field | Required? | Meaning |
| --- | --- | --- |
| `sublattice` | ✓ | Sublattice the potential sits on. |
| `coefficient` | ✓ | Onsite energy expression. |
| `spin_channels` | ✗ | Restrict to `[up]` / `[down]` (spinful models only). |

```{code-block} yaml
- kind: onsite
  sublattice: A
  coefficient: "+M"
```

### `density_density`

A density–density term adds a two-body interaction (and makes the model interacting, so analytic runs use many-body exact diagonalization). It has two variants:

- **Onsite**—omit `from`/`to`/`offsets`. Optionally set `sublattice` to restrict it; otherwise it applies to every site. Requires `spin=2`, coupling the up and down densities on each site (the Hubbard $U$).
- **Bond**—set all of `from`, `to`, and `offsets` to couple densities across a bond.

| Field | Required? | Meaning |
| --- | --- | --- |
| `coefficient` | ✓ | Interaction strength expression. |
| `sublattice` | ✗ | Onsite variant only: restrict to one sublattice. |
| `from` / `to` / `offsets` | Bond only | All three required together for the bond variant. |

```{code-block} yaml
- kind: density_density
  coefficient: "U"
```

## Runtime References

Optimizer, mapper, and ansatz `kwargs` may reference values that are only known at run time using an `@<name>` string. QBP substitutes the live value when it builds the object:

```{code-block} yaml
optimizer:
  type: SPSA
  kwargs:
    maxiter: "@max_iters"

ansatz:
  type: excitation_preserving
  kwargs:
    mode: fsim
    entanglement: linear
    reps: "@n_layers"
  initial_state_prefix: hartree_fock
```

The available runtime names depend on the section: the optimizer sees `max_iters`; the mapper sees `n_sites`, `spin`, `n_occ`, and `num_particles`; the ansatz sees `n_qubits`, `n_layers`, `n_occ`, `spin`, and `n_sites`. See [Optimizers](../components/optimizers.md), [Mappers](../components/mappers.md), and [Initial Simulation States](../components/initial-simulation-states.md) for the supported `type` values.

## Band Structure

If a model has no `spin_channels` on any term, QBP automatically derives a Bloch Hamiltonian from the term list, so band-structure runs work out of the box (this is why `haldane` supports band structure but `kane-mele`, which uses `spin_channels`, does not). Supplying `lattice_vectors` and `sublattice_positions` makes those Bloch phases geometrically correct. To override the automatic derivation, give an explicit `bloch_hamiltonian` block with a `shape` of `[sites_per_cell, sites_per_cell]`, optional `let` intermediates, and `entries` keyed by `"row,col"` expressions in the momentum names (`k`, or `kx`/`ky`/`kz`).

## Annotated Example: `ssh.yaml`

The built-in SSH model is the minimal complete spec—two sublattices, two hopping terms, and a quantum-method stack:

```{code-block} yaml
name: ssh                        # registry key
display_name: SSH                # label for plots / console
spin: 1                          # spinless
n_dims: 1                        # one-dimensional chain
lattice_shape: [Lx]              # single extent, named Lx
sites_per_cell: 2                # bipartite: two sites per cell
sublattices: [A, B]              # their names
parameters:
  t1: {label: "t_1"}             # intra-cell hopping
  t2: {label: "t_2"}             # inter-cell hopping

terms:
  # intra-cell bond (A <-> B at offset 0)
  - kind: hopping
    from: A
    to: B
    offsets: [[0]]
    coefficient: "-t1"
    hermitian_partner: true      # adds the B -> A partner automatically

  # inter-cell bond (B in cell c -> A in cell c+1)
  - kind: hopping
    from: B
    to: A
    offsets: [[1]]
    coefficient: "-t2"
    hermitian_partner: true

optimizer:                       # VQE classical optimizer
  type: SPSA
  kwargs:
    maxiter: "@max_iters"        # filled in from the run's method_params

mapper:                          # fermion-to-qubit mapping
  type: JordanWignerMapper
  kwargs: {}

ansatz:                          # VQE trial circuit
  type: excitation_preserving
  kwargs:
    mode: fsim
    entanglement: linear
    reps: "@n_layers"
  initial_state_prefix: hartree_fock
```

Everything below `terms` is optional; drop the `optimizer`/`mapper`/`ansatz` blocks and the model still runs analytically and falls back to QBP's default quantum-method stack.
