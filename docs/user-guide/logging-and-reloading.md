# Logging and Reloading

Since most runs will be expensive, QBP is built to compute a sweep once and reload it forever after. Passing `log_path` to [`qbp.run`](../api/runners.md) writes the full result to a JSON file; [`qbp.load_result`](../api/runners.md) reads that file back into the same [`RunResult`](../api/results.md) you would have gotten from the run itself. Reloading is instantaneous, so once a sweep is on disk you never pay for it again—you re-plot, re-analyze, and compare against it as often as you like.

## Writing a Run to Disk

`log_path` is the exact file to write. Both the directory and the filename are yours to choose; QBP creates any missing parent directories for you:

```{code-block} python
qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.DMRG],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.1),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.0},
    log_path="runs/haldane-ideal.json",
)
```

QBP imposes no naming scheme and no fixed output location—the full path is yours, so you can organize runs however suits your project and drop them anywhere on disk. `plot_path` works the same way for figures, letting you place the JSON and the plot independently.

## What's in the File

The log is a single JSON object—the run's complete state, not just the numbers:

```{code-block} json
{
    "type": "run",
    "methods": ["analytic", "vqe", "dmrg"],
    "backend": "ideal",
    "plot_format": "3d",
    "band_structure": false,
    "observable": "E",
    "extremum": "min",
    "x_param": "n_occ",
    "y_param": "t2",
    "x_values": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "y_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "parameters": {
        "model": "haldane",
        "lattice": [2, 2],
        "model_params": {"t1": 1.0, "phi": 0.7853981633974483, "M": 0.0},
        "method_params": {
            "vqe": {"iters": 10000, "layers": 5, "reps": 10},
            "dmrg": {"nsweeps": 4, "maxdims": "20,50,100,200", "cutoff": 1e-9}
        }
    },
    "result": {
        "analytic": {"1": {"0": -3.00, "1": -3.42, "2": -3.85, "3": -4.27, "4": -4.70}},
        "vqe":      {"1": {"0": -2.85, "1": -3.23, "2": -3.72, "3": -4.04, "4": -4.47}},
        "dmrg":     {"1": {"0": -3.00, "1": -3.42, "2": -3.85, "3": -4.27, "4": -4.70}}
    }
}
```

The top level records the sweep shape (`x_param` / `y_param` and their sampled `x_values` / `y_values`), the methods present, the backend, and the observable. The `parameters` block captures everything needed to reproduce the run—the model, lattice, fixed `model_params`, and the per-method `method_params`. The `result` block holds the computed values, nested by method and then indexed by grid position: `result[method][ix][iy]` for a two-dimensional sweep, or `result[method][ix]` for a one-dimensional one. It is abbreviated above to one grid column (`ix` of `1`, i.e. $N_\text{occ} = 1$) and its first five $t_2$ values; a full run carries one entry per point in `x_values` × `y_values`.

### Sidecar files for parallel runs

Quantum sweeps run their cells in parallel, and long benchmarks can take hours, so an expensive run also writes two sidecar files next to your `log_path`. A `.raw-data.json` file holds the un-reduced per-cell data (individual repetitions, not just the reduced scalar), refreshed as cells complete. A `.progress.jsonl` journal appends one line per finished cell, which is what lets a crashed or sharded run resume without redoing completed work. Cheap analytic-only runs skip both and write just the summary. You generally never touch these files directly—they exist so that a run interrupted at cell 900 of 1000 doesn't start over from zero.

## Reloading Instead of Recomputing

`load_result` turns a log file back into a `RunResult`, skipping the computation entirely:

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
result.plot()
```

The reloaded object is a full `RunResult`: its `grids`, axes, and metadata are all present, and `.plot()` redraws the figure. This is the intended workflow for anything slow—compute a noisy sweep once with `log_path` set, then reload it whenever you want to re-plot or [analyze the grids](results-and-plotting.md) without waiting on the queue again. The command line exposes the same shortcut with `qbp plot PATH`, which loads a log and renders it (see [CLI and Console](cli-and-console.md)).

`load_result` expects a current-format log. If you hand it a file that isn't a `run` log—an older or unrelated JSON—it raises a `ValueError` telling you to regenerate the run with the current API rather than failing cryptically later.

## Console Logging

QBP configures [loguru](https://loguru.readthedocs.io/) automatically the moment you `import qbp`; the `setup_logging` routine runs at import time and installs two colorized handlers on standard output. `INFO` messages—one per computed cell, reporting the parameters and the resulting value—print in white with a timestamp, so a running sweep streams its progress live. `DEBUG` messages, such as per-iteration VQE energies and per-bit IQPE phases, print dimmed for when you need to see inside a single method. Nothing is written to a log file by default; the run's data goes to your `log_path`, while loguru handles the human-readable progress stream.

If you'd rather silence that stream—inside a notebook, say, or a script that has its own output—remove loguru's handlers before running:

```{code-block} python
from loguru import logger
logger.remove()
```
