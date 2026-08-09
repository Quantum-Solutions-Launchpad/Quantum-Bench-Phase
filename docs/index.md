# QuantumBenchPhase (QBP)

QBP is a library for the generation and benchmarking of quantum phase diagrams of topological and correlated lattice models as well as [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) problem Hamiltonians.

QBP supports six built-in tight binding models and offers an easy API to define your own custom Hamiltonians. Once you choose a model, you can easily run analytic computations generate phase diagrams sweeping over different parameters for a host of different observables, as well as benchmark variational quantum simulation techniques. Beyond its built-in models, QBP can also ingest pre-mapped qubit Hamiltonians directly from the [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) dataset, injecting them into the same simulation pipeline so you can sweep and benchmark those problem Hamiltonians exactly as you would a built-in model.

The library is packaged as both a Python API and a command-line interface. Both support the same features, so use the one you're more comfortable with.

```{toctree}
:caption: Getting Started
:maxdepth: 1

getting-started/installation
getting-started/quickstart
getting-started/concepts
```

```{toctree}
:caption: User Guide
:maxdepth: 1

user-guide/workflow
user-guide/analytic-runs
user-guide/performing-simulation
user-guide/incorporating-quantum-hardware
user-guide/results-and-plotting
user-guide/logging-and-reloading
user-guide/cli-and-console
```

```{toctree}
:caption: Models
:maxdepth: 1

models/catalog
models/custom-python
models/custom-yaml
models/tight-binding-builder
```

```{toctree}
:caption: Components
:maxdepth: 1

components/observables
components/initial-simulation-states
components/mappers
components/optimizers
```

```{toctree}
:caption: API
:maxdepth: 1

api/index
```

```{toctree}
:caption: More Examples
:maxdepth: 1

more-examples/band-structure
more-examples/boundary-conditions
more-examples/mitigation/overview
```

```{toctree}
:caption: Project
:maxdepth: 1

contributing
```
