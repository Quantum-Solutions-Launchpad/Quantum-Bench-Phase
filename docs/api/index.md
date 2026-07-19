# API Reference

This section documents the full public Python API of QBP. The surface divides into a few clusters: you **define or fetch a model**, **run** one or more simulation methods over a parameter sweep, and then **inspect and plot** the result. Around that core sit helpers for pulling in [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) problem Hamiltonians, for real-space and open-boundary analysis of a single Hamiltonian, and for model-specific physics investigations. Everything below is importable directly from `qbp` (e.g. `from qbp import Model, Method, run`); the same features are available through the [command-line interface](../user-guide/cli-and-console.md).

Most users will only need `qbp.run`, a `Method`, and a model name. The rest of the surface is there when you want to build custom models, script the registry, or dig into real-space structure.

## Models and the Registry

Define a model—declaratively or in Python—and manage the name-keyed registry that [`run`](runners.md) looks models up in.

| Object | Summary |
| --- | --- |
| {py:class}`~qbp.Model` | A tight-binding lattice model bundling its Hamiltonians, quantum-method stack, and observables. |
| {py:class}`~qbp.Observable` | A scalar (or per-band) quantity evaluated from a diagonalized Hamiltonian. |
| {py:func}`~qbp.build_tight_binding_model` | Build a `Model` from the tight-binding schema using keyword arguments instead of a YAML file. |
| {py:func}`~qbp.register_model` | Register an in-memory `Model` so it can be driven by name. |
| {py:func}`~qbp.register_model_from_file` | Load a YAML model from disk, register it, and persist the file. |
| {py:func}`~qbp.get_model` | Fetch a registered `Model` by name. |
| {py:func}`~qbp.remove_model` | Unregister a custom model and delete its YAML file. |

## Running Simulations

Choose the technique(s), then launch a sweep or estimate its hardware cost.

| Object | Summary |
| --- | --- |
| {py:class}`~qbp.Method` | The simulation techniques (`ANALYTIC`, `VQE`, `IQPE`, `DMRG`) `run` can execute. |
| {py:func}`~qbp.run` | Run one or more methods over a parameter sweep; the main entry point. |
| {py:func}`~qbp.estimate` | Estimate the QPU-seconds an equivalent `run` on real hardware would cost. |

## Results and Plotting

Everything a run returns, plus ways to reload and re-plot it.

| Object | Summary |
| --- | --- |
| {py:class}`~qbp.RunResult` | A completed sweep's data and reusable plot, returned by `run`. |
| {py:func}`~qbp.load_result` | Reload a `RunResult` from a saved JSON log without re-running. |
| {py:func}`~qbp.plot_diff` | Plot a quantum method's signed error relative to the analytic surface. |

## HamLib Problem Hamiltonians

Ingest pre-mapped qubit Hamiltonians from the HamLib dataset and feed them into the same pipeline via `run(qubit_operator=...)`.

| Object | Summary |
| --- | --- |
| {py:func}`~qbp.list_hamlib_keys` | List every Hamiltonian key in a HamLib HDF5 source (file, zip, or URL). |
| {py:func}`~qbp.load_hamlib_operator` | Load one HamLib Hamiltonian as a Qiskit `SparsePauliOp`. |

## Real-Space and Boundary Analysis

Tools for the open-boundary, single-Hamiltonian diagnostics: site coordinates, quantum-dot geometries, confinement potentials, and edge-state metrics.

| Object | Summary |
| --- | --- |
| {py:func}`~qbp.real_space_positions` | One xy coordinate per real-space site, with spin channels collapsed. |
| {py:class}`~qbp.RealSpaceStateResult` | A single real-space eigenstate, its energy, and its site densities. |
| {py:func}`~qbp.geometry_projection` | Select the sites falling inside a rectangle or disk geometry. |
| {py:class}`~qbp.GeometryProjection` | The set of sites/orbitals a geometry retains from a flake. |
| {py:func}`~qbp.apply_geometry_to_hamiltonian` | Restrict a Hamiltonian to a geometry's retained orbitals. |
| {py:func}`~qbp.soft_dot_potential` | Smooth radial confinement potential $V(r)$ evaluated per site. |
| {py:func}`~qbp.apply_profiles_to_hamiltonian` | Add the soft-dot confinement potential to a Hamiltonian's diagonal. |
| {py:func}`~qbp.edge_mask_from_missing_bonds` | Flag sites that lost connectivity relative to a periodic reference. |
| {py:func}`~qbp.edge_participation_all` | Per-eigenstate probability sitting on the edge sites. |
| {py:func}`~qbp.inverse_participation_ratio_all` | Per-eigenstate inverse participation ratio (localization). |
| {py:class}`~qbp.EdgeSpectrumResult` | The boundary-participation spectrum of an open-boundary Hamiltonian. |

## Investigations

Pluggable, model-specific modifications of the Hamiltonian requiring more complex programmatic extensions than the base API allows.

| Object | Summary |
| --- | --- |
| {py:class}`~qbp.Investigation` | Base class for a model-specific Hamiltonian study. |
| {py:func}`~qbp.build_investigation` | Resolve an investigation name or instance into an `Investigation`. |
| {py:class}`~qbp.SemenoffMass` | Radial Semenoff-mass interface investigation for Haldane-like A/B lattices. |
| {py:func}`~qbp.radial_mass_values` | Per-site target mass from a radial (step or tanh) interface profile. |

## Exceptions

| Object | Summary |
| --- | --- |
| {py:exc}`~qbp.ModelCapabilityError` | Raised when a `Model` is asked for a capability it does not implement. |

```{toctree}
:hidden:
:maxdepth: 1

model
registry
yaml-builder
method
runners
results
hamlib
spatial-analysis
investigations
exceptions
```
