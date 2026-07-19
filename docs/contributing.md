# Contributing

QBP is an open-source library built to be extended as needed to supply new physics or benchmarking needs. Most contributions fall into one of a few shapes—a new **model**, a new **observable**, a new simulation **method**, or a new physics **investigation**—and each has a small, well-defined contract so your addition fully integrates into [`qbp.run`](api/runners.md) and the CLI with no changes to the orchestration. This page covers the development setup and walks through each kind of contribution.

## Repository Layout

The library lives under `qbp/`, with one concern per module (all private, re-exported through `qbp/__init__.py`):

| Path | What's there |
| --- | --- |
| `qbp/_model.py` | The `Model` and `Observable` classes. |
| `qbp/_yaml_model.py` | The declarative YAML schema and `build_tight_binding_model`. |
| `qbp/models/*.yaml` | The six built-in models. |
| `qbp/_method.py` | The `Method` enum and the `SimulationMethod` framework. |
| `qbp/_analytic.py`, `_vqe.py`, `_iqpe.py`, `_dmrg.py` | The four built-in methods, one per module. |
| `qbp/_investigation.py` | The `Investigation` framework; concrete studies live alongside (e.g. `_semenoff_mass.py`). |
| `qbp/_run.py` | The orchestrator: sweep resolution, parallelism, logging, plotting. |
| `qbp/_cli.py` | The command-line interface. |
| `docs/` | This documentation (Sphinx + MyST). |

## Development Setup

Install the package in editable mode so your changes are picked up without reinstalling:

```{code-block} console
$ git clone *****<github link>*****
$ cd qbp
$ pip install -e .
```

Optional extras cover parallelism and hardware backends: `pip install -e ".[parallel]"` for multi-core sweeps, `".[iqm]"` for IQM devices, and `".[gpu-cuda]"` / `".[gpu-torch]"` for GPU-accelerated diagonalization. The DMRG method additionally needs a Julia toolchain with `ITensorMPS`; see [Performing Simulation](user-guide/performing-simulation.md).

To build the docs you also need the Sphinx toolchain, which is not a package dependency:

```{code-block} console
$ pip install sphinx myst-parser jupyter-sphinx sphinx-autodoc-typehints sphinx-copybutton shibuya
```

## Docstring Style

Docstrings use the [numpydoc](https://numpydoc.readthedocs.io/) convention (rendered by `sphinx.ext.napoleon`), with `Parameters`, `Returns`, and `Raises` sections. Keep the first line a one-sentence summary—the [API reference](api/index.md) index table is built from those summaries. Public objects must be re-exported from `qbp/__init__.py` and listed in `__all__` so they appear in the reference.

## Adding a Built-In Model

The fastest route is a declarative [YAML model](models/custom-yaml.md): drop a spec into `qbp/models/`, and it is discovered and registered automatically at import under its `name`. To contribute one, add the `.yaml` file there and the built-in list in `qbp/_registry.py`. For a Hamiltonian that isn't a sum of standard hopping/onsite/density–density terms, define it in Python instead (see [Defining a Model in Python](models/custom-python.md)). Either way, add a page under `docs/models/` documenting it.

## Adding an Observable

An [`Observable`](components/observables.md) is a scalar (or per-band) quantity computed from a diagonalized Hamiltonian. To make one available to every model, add it to the built-in defaults assembled in `Model.__init__` (`qbp/_model.py`); to add one to a single model, pass it through the constructor's `observables` argument. See [Observables](components/observables.md) for the evaluator contract and a worked example.

## Adding a Built-In Method

A simulation method is a {py:class}`~qbp.Method` enum member plus a `SimulationMethod` subclass. The base class (`qbp/_method.py`) owns everything shared—sweep resolution, parallelism, logging, plotting—so a new method only declares its parameters and implements the per-cell compute hooks it supports. There are three steps.

**1. Add the enum member.** In `qbp/_method.py`, add your technique to the `Method` enum:

```{code-block} python
class Method(Enum):
    ANALYTIC = "analytic"
    VQE = "vqe"
    IQPE = "iqpe"
    DMRG = "dmrg"
    CUSTOM = "custom"    # your new method
```

**2. Write the method class** in its own module, e.g. `qbp/_custom_method.py`. Declare tunable parameters with `ParamSpec`—the single source of truth that drives both `method_params` validation and the auto-generated `--custom-<name>` CLI flags—set the `SUPPORTS_*` flags for the run kinds you handle, and implement the matching compute hooks. Each hook returns a dict the base class reduces to the plotted value:

```{code-block} python
from qbp._method import Method, ParamSpec, SimulationMethod, register_method


@register_method
class CustomMethod(SimulationMethod):
    METHOD = Method.CUSTOM
    LABEL = "Custom"
    PARAM_SPECS = [
        ParamSpec("layers", int, 1, "Number of circuit layers", metavar="N"),
        ParamSpec("iters", int, 100, "Optimizer iterations", metavar="N"),
    ]
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = False
    SUPPORTS_OPERATOR = True

    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        energy = ...  # run your algorithm for this sweep cell
        return {"repetitions": [float(energy)]}
```

The compute hooks are `compute_cell` (real-space lattice sweeps), `compute_bloch_cell` (momentum/band sweeps), and `compute_operator_cell` (the HamLib qubit-operator path); the orchestrator only calls the ones whose `SUPPORTS_*` flag is set. Return `{"value": ...}` for a single number, `{"bands": [...]}` for a band list, or `{"repetitions": [...]}` when you have several stochastic repetitions (the base `reduce` picks the one closest to the analytic reference). Override `reduce`, add `estimate_*` hooks for [resource estimation](user-guide/incorporating-quantum-hardware.md), or set `MITIGATION_CLASS` for error mitigation as needed—see the existing `_vqe.py` and `_iqpe.py` for full examples.

**3. Register it for discovery.** The `@register_method` decorator adds the class to the registry, but the module must be imported for the decorator to run. Add that import to `_ensure_registered` in `qbp/_method.py`:

```{code-block} python
def _ensure_registered() -> None:
    import qbp._analytic  # noqa: F401
    import qbp._vqe       # noqa: F401
    import qbp._iqpe      # noqa: F401
    import qbp._dmrg      # noqa: F401
    import qbp._custom_method  # noqa: F401  # your new method
```

Your method is now selectable as `method=[Method.CUSTOM]` (or `"custom"`), tunable via `method_params`, and exposed on the CLI—no changes to `qbp.run` required.

## Adding an Investigation

An {py:class}`~qbp.Investigation` is a pluggable, model-specific modification of the single-particle Hamiltonian—the physics analogue of a method. Where a method chooses the *solver*, an investigation chooses the *physics being probed*, so it is the right home for a study that only makes sense for certain models (like the bundled [`SemenoffMass`](api/investigations.md) interface, which requires an A/B lattice). Unlike `Method`, investigations use a string-keyed registry with no closed enum, so adding one is two steps.

**1. Write the investigation class** in its own module, e.g. `qbp/_my_study.py`. Set `NAME` (its registry key) and `LABEL`, declare parameters with `ParamSpec`, gate the study on model capability in `check_model` (raising {py:exc}`~qbp.ModelCapabilityError` when unmet), and modify the projected Hamiltonian in `apply`:

```{code-block} python
import numpy as np
from qbp._method import ParamSpec
from qbp._model import ModelCapabilityError
from qbp._investigation import Investigation, register_investigation


@register_investigation
class MyStudy(Investigation):
    NAME = "my_study"
    LABEL = "My study $\\Delta(r)$"
    PARAM_SPECS = [
        ParamSpec("strength", float, help="Perturbation strength."),
    ]

    def check_model(self, model) -> None:
        if model.spin != 1:
            raise ModelCapabilityError(
                f"investigation '{self.NAME}' requires a spinless model."
            )

    def apply(self, H, model, projection, model_params) -> np.ndarray:
        out = H.copy()
        # add this study's terms to the projected Hamiltonian
        return out
```

`apply` receives the geometry-projected Hamiltonian `H`, the `model`, the [`GeometryProjection`](api/spatial-analysis.md) (site indices and positions), and the resolved `model_params`, and returns the modified `H`.

**2. Register it for discovery.** As with methods, add the module import to `_ensure_registered` in `qbp/_investigation.py` so the `@register_investigation` decorator runs:

```{code-block} python
def _ensure_registered() -> None:
    import qbp._semenoff_mass  # noqa: F401
    import qbp._my_study       # noqa: F401  # your new investigation
```

Callers then select it with `run(investigation="my_study", investigation_params={"strength": 0.5})`, and it also appears as a `--investigation` choice on the CLI. Investigations apply to real-space analytic runs and diagnostics.

## Building the Documentation

The docs are Sphinx + MyST. Rebuild them with:

```{code-block} console
$ make -C docs html
```

The build runs every `{jupyter-execute}` block, so new executable examples run at build time—keep them cheap (small lattices, analytic runs where possible). The build treats warnings as errors (`-W`), so unresolved cross-references or broken examples fail it; fix any warning it reports. The rendered site lands in `docs/_build/html/`.
