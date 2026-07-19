# Qubit Mappers

Before a quantum method can run, the fermionic Hamiltonian has to be rewritten in terms of qubit operators. A **qubit mapper** performs that fermion-to-qubit encoding, turning creation/annihilation operators $c_i^\dagger, c_i$ into Pauli strings. The choice affects how many qubits the circuit needs and how heavy the resulting Pauli operators are, which in turn drives circuit depth and measurement cost. QBP wraps the mappers from `qiskit_nature.second_q.mappers` so you can pick one per model.

## Supported Mappers

| `type` | Trade-off |
| --- | --- |
| `JordanWignerMapper` | One spin-orbital → one qubit. The default. |
| `ParityMapper` | Stores parities rather than occupations; supports a two-qubit reduction. |
| `BravyiKitaevMapper` | Balances locality against qubit count via a tree encoding. |

**Jordan–Wigner** is the most direct encoding: spin-orbital $i$ maps to qubit $i$, so the qubit count equals the number of spin-orbitals (`n_sites × spin`). Its cost is that enforcing fermionic antisymmetry attaches a string of $Z$ operators to each term, so a single hopping term can become a long Pauli string whose weight grows with the distance between the orbitals. It is the natural default and the easiest to reason about.

**Parity** stores the cumulative parity of the occupation instead of the occupation itself, which moves the $Z$ string to the other end of the operator. Its practical advantage is that, given the number of particles, it admits a two-qubit reduction: two qubits carry redundant symmetry information and can be removed, shrinking the register. Supply `num_particles` to enable that reduction.

**Bravyi–Kitaev** uses a binary-tree encoding that stores a mix of occupation and parity information, so both the occupation and the parity of any orbital can be read from $O(\log N)$ qubits. This caps the weight of the Pauli strings at logarithmic in the system size rather than linear, trading Jordan–Wigner's simplicity for shorter operators on larger systems.

## Configuring a Mapper

A mapper spec is a `type` plus a `kwargs` dict forwarded to the Qiskit mapper class. In a [YAML model](../models/custom-yaml.md):

```{code-block} yaml
mapper:
  type: JordanWignerMapper
  kwargs: {}
```

`kwargs` values written as `@<name>` are **runtime references**, filled in when the mapper is built. The mapper sees `n_sites`, `spin`, `n_occ`, and `num_particles` (a `(n_up, n_down)` tuple derived from `n_occ` and `spin`). This is how you feed the parity mapper the particle count it needs for the two-qubit reduction:

```{code-block} yaml
mapper:
  type: ParityMapper
  kwargs:
    num_particles: "@num_particles"
```

## Defaults and Where to Set It

If a model specifies no mapper, QBP uses a plain `JordanWignerMapper`, so the qubit count matches the number of spin-orbitals. The mapper is a property of the model:

- **YAML models**—add the `mapper` block shown above; see the [YAML schema](../models/custom-yaml.md).
- **Python models**—pass a `get_mapper` function `(n_sites, spin, n_occ) -> QubitMapper` to the [`Model`](../api/model.md) constructor.

The same mapper is used consistently across a model's quantum methods, so the encoding you choose applies to VQE and IQPE alike, as well as to any observables that are measured as qubit operators.
