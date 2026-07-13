# Programmatic Tight-Binding Builder

[`build_tight_binding_model`](../api/yaml-builder.md) is the in-memory equivalent of a [YAML model file](custom-yaml.md): it takes exactly the same schema, but as Python keyword arguments instead of a file, and returns a ready-to-register [`Model`](../api/model.md). It runs the same validation as the YAML path, so the two are interchangeable—reach for the builder when the *structure* of a model is itself something you want to compute or sweep in Python, rather than writing (and re-writing) YAML by hand.

## Signature

```{eval-rst}
.. autofunction:: qbp.build_tight_binding_model
   :noindex:
```

The arguments mirror the [YAML top-level keys](custom-yaml.md#top-level-keys) one-for-one. `terms` is a list of dicts, each shaped like a YAML term (note the reserved word `from` is fine as a dict key); `parameters` maps each name to `{"label": "<LaTeX>"}`; and `optimizer`, `mapper`, `ansatz`, `bloch_hamiltonian`, `lattice_vectors`, `sublattice_positions`, and `observables` are the same optional blocks, passed as dicts. The function only *builds* the model—call [`register_model`](../api/registry.md) to make it available to [`qbp.run`](../api/runners.md) by name.

## Worked Example

The following reproduces `qbp/models/ssh.yaml` exactly, then registers and runs it:

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

model = qbp.build_tight_binding_model(
    name="ssh_prog",
    display_name="SSH (programmatic)",
    spin=1,
    n_dims=1,
    lattice_shape=["Lx"],
    sites_per_cell=2,
    sublattices=["A", "B"],
    parameters={"t1": {"label": "t_1"}, "t2": {"label": "t_2"}},
    terms=[
        {"kind": "hopping", "from": "A", "to": "B",
         "offsets": [[0]], "coefficient": "-t1", "hermitian_partner": True},
        {"kind": "hopping", "from": "B", "to": "A",
         "offsets": [[1]], "coefficient": "-t2", "hermitian_partner": True},
    ],
    optimizer={"type": "SPSA", "kwargs": {"maxiter": "@max_iters"}},
    mapper={"type": "JordanWignerMapper", "kwargs": {}},
    ansatz={
        "type": "excitation_preserving",
        "kwargs": {"mode": "fsim", "entanglement": "linear", "reps": "@n_layers"},
        "initial_state_prefix": "hartree_fock",
    },
)

qbp.register_model(model)

result = qbp.run(
    model="ssh_prog",
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

## When to Use the Builder

Because the terms are ordinary Python data, you can generate them in a loop and register a *family* of models parameterized by their structure. For example, comparing SSH chains with different numbers of sites per cell, or generating a model per lattice geometry:

```{code-block} python
import qbp

for reps in (2, 3, 4):
    terms = [
        {"kind": "hopping", "from": f"s{i}", "to": f"s{i + 1}",
         "offsets": [[0]], "coefficient": "-t", "hermitian_partner": True}
        for i in range(reps - 1)
    ]
    terms.append({"kind": "hopping", "from": f"s{reps - 1}", "to": "s0",
                  "offsets": [[1]], "coefficient": "-t", "hermitian_partner": True})
    model = qbp.build_tight_binding_model(
        name=f"chain_{reps}",
        display_name=f"Chain (n={reps})",
        spin=1, n_dims=1, lattice_shape=["Lx"],
        sites_per_cell=reps,
        sublattices=[f"s{i}" for i in range(reps)],
        parameters={"t": {"label": "t"}},
        terms=terms,
    )
    qbp.register_model(model)
```

To summarize, you should use a hand-written [YAML file](custom-yaml.md) when a model is a fixed artifact you want to version and share—`register_model_from_file` persists it under `qbp/models/`. You should use the builder when the model definition is dynamic, or when you'd rather keep everything in one Python script. For Hamiltonians that aren't expressible as a sum of hopping / onsite / density–density terms, drop to the full [Python `Model` constructor](custom-python.md).
