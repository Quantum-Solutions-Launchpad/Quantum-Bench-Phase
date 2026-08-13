# Incorporating Quantum Hardware

Ideal simulation asks whether an algorithm *can* recover the right answer. Real hardware asks whether it still can once gate errors, readout errors, and decoherence get in the way. QBP routes the quantum methods through a noise model or onto physical hardware with a single keyword: `backend`. Everything else about the run—the sweep, the methods, the `method_params`—stays exactly the same, so a noisy run is a one-argument change away from the ideal run it mirrors.

The `backend` argument accepts four kinds of value:

- **`None`** — the default. VQE and IQPE execute on a noise-free `AerSimulator`. This is the noise-free case covered under [Performing Simulation](performing-simulation.md).
- **A fake backend** — a Qiskit *fake* device (an object, or a name like `"FakeSherbrooke"`). QBP builds an `AerSimulator` from that device's recorded noise model, so you get realistic noise locally without touching a queue.
- **A real IBM device** — a device name (`"ibm_brisbane"`), the string `"least_busy"`, or an `IBMBackend` object. Circuits run on hardware through Qiskit Runtime.
- **A real IQM device** — `"iqm_emerald"`, `"iqm_garnet"`, or `"iqm_sirius"`. Circuits run on IQM hardware through the IQM Resonance cloud.

## Local Noisy Simulation

The quickest way to see noise is a fake backend. Here we sweep the Haldane model's bands across the Brillouin zone with the analytic reference, VQE, and IQPE at once, passing `backend="FakeSherbrooke"` to route the circuits through a snapshot of IBM's Sherbrooke device. A momentum-space sweep is a good first noisy benchmark: each $\mathbf{k}$ point is an independent two-level Bloch Hamiltonian, so the circuits stay shallow enough that the run finishes in minutes and any deviation you see is genuinely the device's noise rather than an unconverged optimization.

```{jupyter-execute}
:hide-code:

import io
import os
import sys
from pathlib import Path
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()

import qbp
from qbp import Method

def _find_data_dir() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "docs" / "_data"
        if candidate.is_dir():
            return candidate
        candidate = base / "_data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("docs/_data not found relative to cwd")

_DATA_DIR = _find_data_dir()

_real_run = qbp.run

def _patched_run(*args, **kwargs):
    """Serve the pre-computed noisy sweep from docs/_data instead of
    re-running it against FakeSherbrooke at documentation-build time."""
    methods = kwargs.get("method") or []
    names = {getattr(m, "value", m) for m in methods}
    if names & {"vqe", "iqpe"}:
        result = qbp.load_result(str(_DATA_DIR / "simulated-noisy-band-E-kx-vs-ky.json"))
        result.plot(hide_plot=kwargs.get("hide_plot", False))
        return result
    return _real_run(*args, **kwargs)

qbp.run = _patched_run
```

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    x_param="kx",
    y_param="ky",
    x_range=(-math.pi, math.pi, math.pi / 8),
    y_range=(-math.pi, math.pi, math.pi / 8),
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 4, "M": 0.0},
    method_params={
        Method.VQE: {"iters": 400, "layers": 4, "reps": 5},
        Method.IQPE: {"time": 0.2, "trot": 5, "iters": 8, "reps": 20},
    },
    backend="FakeSherbrooke",
)
```

The analytic result is the pair of exact Bloch bands—it never sees the backend—and the quantum markers track its lower band. Both stay close on a problem this shallow: VQE lands within about $0.02$ of the exact band on average and IQPE within about $0.03$, with IQPE scattering more because its bit-by-bit phase readout has no variational averaging to absorb a bad shot. Any fake backend shipped with `qiskit-ibm-runtime` works here; pass the class name (`"FakeSherbrooke"`, `"FakeBrisbane"`, …) or your own Qiskit backend object.

## Running on Real Hardware

Swapping the fake backend for a device name sends the circuits to physical qubits instead. Nothing else about the call changes—here on a small real-space sweep, since queued hardware runs want as few cells as you can get away with:

```{code-block} python
result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    backend="ibm_brisbane",   # or "least_busy" to pick the least-busy device
)
```

Real backends run one cell at a time rather than in parallel, and every cell waits in the provider's queue, so a sweep that finishes in seconds on a simulator can take hours on hardware. Keep hardware runs small: coarse ranges, few `reps`, and a lattice no larger than the device can hold. Use a fake backend to shake out the sweep first, then promote it to hardware once you're confident in the shape.

### IBM Cloud authentication

IBM devices execute through Qiskit Runtime, which needs credentials on the machine running QBP. Save an account once and it persists:

```{code-block} python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="<your API token>",
    instance="<your instance>",
)
```

Alternatively, set the `QISKIT_IBM_TOKEN` and `QISKIT_IBM_INSTANCE` environment variables. If no account is found, QBP raises an error pointing you back to this step rather than silently falling back to a simulator.

## Running on IQM Resonance

IQM's Resonance devices are selected by name and require the IQM extra:

```{code-block} console
$ pip install "qbp[iqm]"
```

With that installed, point `backend` at one of the Resonance devices:

```{code-block} python
result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    backend="iqm_garnet",   # or "iqm_emerald" / "iqm_sirius"
)
```

### IQM Resonance authentication

Generate an API token from the IQM Resonance dashboard (**Dashboard → Generate token**) and export it:

```{code-block} console
$ export IQM_TOKEN=<your token>
```

QBP also reads the token from a `.env` file in the working directory if one is present. As with IBM, a missing token surfaces as an explicit error with these instructions rather than a silent fallback.

The same runtime caveats apply: hardware runs are queued, serialized, and slow. Validate against `FakeSherbrooke` or an IQM ideal run before committing a full sweep to a physical device.
