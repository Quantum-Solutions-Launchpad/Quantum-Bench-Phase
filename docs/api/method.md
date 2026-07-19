# Method

{py:class}`~qbp.Method` enumerates the four simulation techniques {py:func}`~qbp.run` can execute: exact-diagonalization `ANALYTIC`, the quantum algorithms `VQE` and `IQPE`, and the classical tensor-network benchmark `DMRG`. You pass one or a list as `run(method=...)`, and each method's per-run settings go in `method_params`, keyed by the member itself. See [Performing Simulation](../user-guide/performing-simulation.md) for what each method computes and how to tune it.

```{eval-rst}
.. autoclass:: qbp.Method
   :members:
```

## Examples

Pass a list to overlay several methods on one figure. Keeping `Method.ANALYTIC` in the list draws the exact surface as a reference the quantum methods sit on top of:

```{code-block} python
import qbp
from qbp import Method

result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    method_params={
        Method.VQE: {"iters": 100, "layers": 1, "reps": 1},
        Method.IQPE: {"time": 0.1, "trot": 1, "iters": 1},
    },
)
```

Each method reads only its own entry in `method_params`, so VQE and IQPE are tuned independently in a single call. Method names are also accepted as strings (`"analytic"`, `"vqe"`, …) wherever a `Method` is expected.
