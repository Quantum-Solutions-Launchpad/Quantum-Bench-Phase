# Runners

{py:func}`~qbp.run` is the single entry point that drives every simulation. You give it a model, one or more [methods](method.md), and a description of the parameter sweep; it returns a {py:class}`~qbp.RunResult`. {py:func}`~qbp.estimate` mirrors `run`'s signature but, instead of executing, reports the QPU-seconds the equivalent run would cost on real hardware. The [End-to-End Workflow](../user-guide/workflow.md) and [Performing Simulation](../user-guide/performing-simulation.md) guides put these in context.

```{eval-rst}
.. autofunction:: qbp.run
```

The sweep axes decide the kind of figure. A one-axis sweep (`x_param`) draws a line; two axes (`x_param` + `y_param`) draw a 3D surface or, with `heatmap=True`, a heatmap. Momentum axes (`k`, `kx`/`ky`) trigger a [band-structure](../more-examples/band-structure.md) run, and the lattice axes (`Lx`/`Ly`) or `eigenstate` trigger the real-space diagnostics.

A minimal analytic sweep over one parameter:

```{code-block} python
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC],
    lattice=(8,),
    x_param="t2",
    x_range=(0.0, 2.0, 0.02),
    model_params={"t1": 1.0},
    observable="gap",
)
```

The same sweep with the quantum methods on the ideal (noise-free) simulator—leave `backend` unset:

```{code-block} python
result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(4,),
    x_param="t2",
    x_range=(0.0, 2.0, 0.1),
    model_params={"t1": 1.0},
    method_params={
        Method.VQE: {"iters": 100, "layers": 1},
        Method.IQPE: {"time": 0.1, "trot": 1, "iters": 1},
    },
)
```

And against a local noise model by naming a fake backend (see [Incorporating Quantum Hardware](../user-guide/incorporating-quantum-hardware.md)):

```{code-block} python
result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC, Method.VQE],
    lattice=(4,),
    x_param="t2",
    x_range=(0.0, 2.0, 0.25),
    model_params={"t1": 1.0},
    backend="FakeSherbrooke",
)
```

```{eval-rst}
.. autofunction:: qbp.estimate
```

Because `estimate` accepts the same arguments as `run`, any real-hardware run can be costed by changing the verb. It requires `backend` to resolve to a real IBM or IQM device and returns the total estimated QPU-seconds. See [Resource Estimation](../user-guide/incorporating-quantum-hardware.md) for the workflow.

```{code-block} python
import qbp
from qbp import Method

seconds = qbp.estimate(
    model="ssh",
    method=[Method.VQE],
    lattice=(4,),
    x_param="t2",
    x_range=(0.0, 2.0, 0.25),
    model_params={"t1": 1.0},
    backend="ibm_brisbane",
)
```
