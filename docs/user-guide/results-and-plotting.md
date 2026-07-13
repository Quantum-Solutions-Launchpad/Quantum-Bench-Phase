# Results and Plotting

Every call to [`qbp.run`](../api/runners.md) returns a [`RunResult`](../api/results.md). It is a plain dataclass that carries the sweep axes, the computed grids, and enough metadata to redraw the figure at any time. Because it is just data, you can hold on to it, inspect it, feed the grids into your own analysis, or re-plot it without recomputing anything.

## Inspecting a Result

Here we load a previously saved Haldane run from disk rather than recomputing it—[`qbp.load_result`](../api/runners.md) reconstructs the exact same `RunResult` (see [Logging and Reloading](logging-and-reloading.md)):

```{jupyter-execute}
:hide-code:

import io
import sys
from pathlib import Path
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()

from IPython import get_ipython
get_ipython().ast_node_interactivity = "none"

def _find_data_dir() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        for name in ("docs/_data", "_data"):
            candidate = base / name
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("docs/_data not found relative to cwd")

_DATA_DIR = _find_data_dir()
```

```{jupyter-execute}
import qbp

result = qbp.load_result(str(_DATA_DIR / "simulated-ideal-3d-n_occ-vs-t2.json"))

print("model:     ", result.model_name)
print("methods:   ", result.methods)
print("axes:      ", result.x_param, "x", result.y_param)
print("observable:", result.observable)
print("backend:   ", result.backend_label)
```

The fields you'll reach for most often are:

- **`model_name`.** The model the run was computed for.
- **`lattice`.** The lattice extents, or `None` for band-structure and operator runs.
- **`x_param` / `y_param`.** The sweep axes. `y_param` is `None` for a one-dimensional sweep.
- **`x_values` / `y_values`.** The sampled grid points along each axis.
- **`methods`.** The methods present in the result, as strings (`"analytic"`, `"vqe"`, `"iqpe"`, `"dmrg"`).
- **`grids`.** A dictionary mapping each method name to its NumPy array of computed values—the raw numbers behind the plot.
- **`observable`.** Which observable was computed. Fixed at run time.
- **`backend_label`.** `"ideal"` for a noise-free run, or the backend's name for a noisy one.
- **`band_structure` / `plot_format`.** Whether this is a momentum-space run, and how it renders (`"2d"`, `"3d"`, or `"heatmap"`).
- **`log_path` / `plot_path`.** Where the run wrote its JSON and figure, if anywhere.

Because `grids` holds ordinary NumPy arrays, computing the VQE error against the analytic baseline is a one-liner:

```{jupyter-execute}
import numpy as np

error = np.abs(result.grids["vqe"] - result.grids["analytic"])
print("max VQE deviation from analytic:", float(error.max()))
```

## Plotting

A `RunResult` knows how to draw itself. `qbp.run` calls `.plot()` for you unless you pass `hide_plot=True`, but you can always call it directly—useful after loading a saved run:

```{jupyter-execute}
result.plot()
```

The signature is:

```{code-block} python
result.plot(
    *,
    output_path=None,   # write the figure to this file
    hide_plot=False,    # suppress the on-screen figure
    hide_legend=False,  # drop the legend
    diff=False,         # also draw method-vs-method difference plots
    diff_format="3d",   # format for the diff plots: "3d", "heatmap", or "bar_2d"
)
```

QBP chooses the figure type from the run itself: a one-dimensional sweep becomes a line plot, a two-dimensional sweep a 3D surface (or a heatmap if the run was created with `heatmap=True`), and a multi-method run draws the reference method as a surface with the others overlaid as markers. Axis labels are derived automatically from the model's parameter labels and the observable's display name, so a sweep over `t2` is labeled $t_2$ and a gap run is labeled $\Delta_\text{gap}$ without any extra configuration.

Two flags shape the figure directly. `hide_legend=True` removes the legend, which declutters dense multi-method surfaces. `diff=True` adds a difference plot for every pair of methods—each shows $E_b - E_a$ where $a$ precedes $b$ in the canonical method order, so quantum methods are differenced against the analytic reference by convention. `diff_format` controls how those difference plots render.

## Saving to a File

Pass `output_path` to write the figure. The file extension picks the format—matplotlib handles PDF, PNG, SVG, and the rest:

```{code-block} python
result.plot(output_path="haldane-sweep.pdf")   # vector PDF, auto-cropped
result.plot(output_path="haldane-sweep.png")   # raster PNG
```

PDF output is trimmed with `pdfcrop` automatically when it's available. You can save and suppress the interactive figure at once by combining `output_path` with `hide_plot=True`, which is the usual pattern in scripts and batch jobs. Passing `plot_path` to `qbp.run` is equivalent to calling `.plot(output_path=...)` on the returned result.
