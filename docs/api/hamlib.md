# HamLib Operators

Beyond its built-in lattice models, QBP can ingest pre-mapped qubit Hamiltonians from the [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) dataset and run them through the same pipeline. These two functions browse and load a HamLib HDF5 source—a local file, a zip archive, or an `http(s)` URL, all handled transparently. The loaded {py:class}`~qiskit.quantum_info.SparsePauliOp` is then handed to {py:func}`~qbp.run` through its `qubit_operator` argument, which sweeps and benchmarks it exactly as it would a built-in model.

```{eval-rst}
.. autofunction:: qbp.list_hamlib_keys
```

```{code-block} python
import qbp

keys = qbp.list_hamlib_keys("electronic_structure.hdf5")
for k in keys[:10]:
    print(k)
```

```{eval-rst}
.. autofunction:: qbp.load_hamlib_operator
```

Pick a key from the listing and load it, then feed it to {py:func}`~qbp.run` via `qubit_operator`:

```{code-block} python
import qbp
from qbp import Method

op = qbp.load_hamlib_operator("electronic_structure.hdf5", key=keys[0])

result = qbp.run(
    method=[Method.VQE],
    qubit_operator=op,
    extremum="min",
)
```

`run` also accepts the HDF5 source (and an optional `select` filter) directly as `qubit_operator`, so you can skip the explicit load when sweeping a family of Hamiltonians; see [Incorporating Quantum Hardware](../user-guide/incorporating-quantum-hardware.md).
