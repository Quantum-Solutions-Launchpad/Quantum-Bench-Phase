# Results

{py:func}`~qbp.run` returns a {py:class}`~qbp.RunResult`: the sweep's data (per-method grids keyed by method name) together with the metadata needed to redraw its figure. {py:func}`~qbp.load_result` rebuilds that same object from a saved JSON log, and {py:func}`~qbp.plot_diff` renders a quantum method's error relative to the analytic surface from such a log. See [Results and Plotting](../user-guide/results-and-plotting.md) and [Logging and Reloading](../user-guide/logging-and-reloading.md) for the wider workflow.

```{eval-rst}
.. autoclass:: qbp.RunResult
   :members:
```

`run` draws the figure for you unless `hide_plot=True`. Call {py:meth}`~qbp.RunResult.plot` to redraw it—for example to save to a specific file after inspecting the numbers, or to switch to the analytic-difference view:

```{code-block} python
import qbp
from qbp import Method

result = qbp.run(
    model="ssh",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(4,),
    x_param="t2",
    x_range=(0.0, 2.0, 0.1),
    model_params={"t1": 1.0},
    hide_plot=True,
)

# the raw numbers, keyed by method name
print(result.grids["analytic"])
print(result.grids["vqe"])

# redraw and save to PDF
result.plot(output_path="ssh_sweep.pdf", hide_plot=True)
```

```{eval-rst}
.. autofunction:: qbp.load_result
```

Pass `log_path` to `run` to write a JSON log, then reload it later to re-plot or inspect without paying to run the sweep again:

```{code-block} python
import qbp

result = qbp.load_result("runs/ssh_sweep.json")
result.plot()
```

```{eval-rst}
.. autofunction:: qbp.plot_diff
```

`plot_diff` reads a run log in which a quantum method was run alongside `Method.ANALYTIC` and plots the signed difference `E_method - E_analytic`, which makes algorithmic and hardware error easy to read off:

```{code-block} python
import qbp

qbp.plot_diff("runs/ssh_sweep.json", method="both", plot_format="heatmap")
```
