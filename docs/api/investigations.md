# Investigations

An {py:class}`~qbp.Investigation` is a pluggable, model-specific modification of the single-particle Hamiltonian—the physics analogue of a [`Method`](method.md). Where a method chooses the *solver*, an investigation chooses the *physics* being probed: a study that only makes sense for certain models, such as a radial Semenoff-mass interface on a Haldane-like A/B lattice. You select one through {py:func}`~qbp.run`'s `investigation` and `investigation_params` arguments—mirroring the `model`/`model_params` and `method`/`method_params` pairings—and it applies to real-space analytic runs and diagnostics.

```{eval-rst}
.. autoclass:: qbp.Investigation
   :members:
```

To add a study of your own, subclass `Investigation`, declare its tunable parameters, gate it on model capability in `check_model`, modify the Hamiltonian in `apply`, and register it. Adding one costs no change to {py:func}`~qbp.run`.

```{eval-rst}
.. autofunction:: qbp.build_investigation
```

`run` calls this internally to resolve whatever you pass as `investigation`—a registered name (with `investigation_params`) or a prebuilt instance—into an `Investigation`.

```{eval-rst}
.. autoclass:: qbp.SemenoffMass
   :members:
   :show-inheritance:
```

The bundled investigation. It imposes a radial mass interface on an A/B lattice, so a topological inner region meets a trivial outer region across a chosen radius. Select it by name and set the base model parameter `M` to `0.0` so the profile supplies the full mass:

```{code-block} python
import qbp
from qbp import Method

result = qbp.run(
    model="haldane-honeycomb",
    method=[Method.ANALYTIC],
    lattice=(12, 12),
    boundary="open",
    x_param="eigenstate",
    investigation="semenoff_mass",
    investigation_params={
        "profile": "radial_tanh",
        "radius": 4.0,
        "inner": -0.3,
        "outer": 0.3,
        "xi": 1.0,
    },
    model_params={"t1": 1.0, "t2": 0.1, "phi": 0.5, "M": 0.0},
)
```

```{eval-rst}
.. autofunction:: qbp.radial_mass_values
```

The standalone helper `SemenoffMass` uses under the hood: given site coordinates, it returns the per-site target mass for a `radial_step` or `radial_tanh` interface. It is exposed so you can compute or plot the profile directly.
