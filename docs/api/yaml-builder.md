# Build Tight-Binding Model

{py:func}`~qbp.build_tight_binding_model` builds a {py:class}`~qbp.Model` from the same declarative tight-binding schema as a [YAML model](../models/custom-yaml.md), but takes the fields as keyword arguments instead of a file. Reach for it when you want a standard sum-of-terms model constructed in Python—no YAML file to manage—while still getting the schema's validation and automatic Bloch-Hamiltonian derivation. For a Hamiltonian that isn't a sum of standard hopping/onsite/density–density terms, use the full [`Model`](model.md) constructor instead. The [programmatic builder guide](../models/tight-binding-builder.md) walks through the fields in depth.

```{eval-rst}
.. autofunction:: qbp.build_tight_binding_model
```

## Examples

The following reproduces the built-in `qbp/models/ssh.yaml` in Python—two sublattices and two hopping terms—then registers it:

```{code-block} python
import qbp

ssh = qbp.build_tight_binding_model(
    name="ssh_py",
    display_name="SSH (built in Python)",
    spin=1,
    n_dims=1,
    lattice_shape=["Lx"],
    sites_per_cell=2,
    sublattices=["A", "B"],
    parameters={"t1": {"label": "t_1"}, "t2": {"label": "t_2"}},
    terms=[
        {"kind": "hopping", "from": "A", "to": "B", "offsets": [[0]],
         "coefficient": "-t1", "hermitian_partner": True},
        {"kind": "hopping", "from": "B", "to": "A", "offsets": [[1]],
         "coefficient": "-t2", "hermitian_partner": True},
    ],
)
qbp.register_model(ssh)
```

The optional `optimizer`, `mapper`, and `ansatz` arguments take the same `{type, kwargs}` dicts as the [YAML blocks](../models/custom-yaml.md#runtime-references); omit them to inherit QBP's default quantum-method stack.
