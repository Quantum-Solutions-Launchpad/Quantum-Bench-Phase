# API Reference

```{eval-rst}
.. todo:: One-paragraph framing of the public API surface (everything in
   :py:data:`quaph.__all__`).
```

```{eval-rst}
.. currentmodule:: quaph

.. autosummary::
   :nosignatures:

   Model
   Observable
   ModelCapabilityError
   get_model
   register_model
   register_model_from_file
   remove_model
   build_tight_binding_model
   run_analytic
   run_simulated_ideal
   run_simulated_noisy
   load_result
   AnalyticResult
   SimulatedResult
```

```{toctree}
:hidden:
:maxdepth: 1

model
observable
registry
yaml-builder
runners
results
exceptions
```
