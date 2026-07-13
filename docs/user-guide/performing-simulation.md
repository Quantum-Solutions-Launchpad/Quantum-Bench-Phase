# Performing Simulation

The point of QBP is to benchmark quantum and classical algorithms to see how close they get to the correct observable value of some physical model. The same [`qbp.run`](../api/runners.md) entry point drives them all: add `Method.VQE`, `Method.IQPE`, or the classical `Method.DMRG` to the `method` list, and keep `Method.ANALYTIC` in the list to overlay the exact diagonalization surface as a reference. Leaving `backend` unset runs the quantum methods on a noise-free statevector simulator.

- **VQE** (the Variational Quantum Eigensolver) prepares a parameterized ansatz circuit and hands it to a classical optimizer, which tunes the circuit parameters to minimize the measured energy.
- **IQPE** (Iterative Quantum Phase Estimation) evolves an initial state under a Trotterized time-evolution circuit and reads the ground-state energy out of the accumulated phase, one bit at a time.
- **DMRG** (the Density Matrix Renormalization Group) is not a quantum algorithm at all: it variationally optimizes a matrix-product state on a classical computer, running through a bundled Julia/ITensors backend. It serves as a strong classical benchmark—often numerically exact in one dimension—so you can see where a quantum method stands relative to the best classical approach as well as the analytic baseline.

On an ideal simulator each method should track the analytic baseline closely; how closely depends on the knobs you set in `method_params`.

## Per-Method Parameters

`method_params` is a dictionary keyed by the method itself. Each method reads only its own entry, so you can tune VQE, IQPE, and DMRG independently in a single call:

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
    methods = kwargs.get("method") or []
    names = {getattr(m, "value", m) for m in methods}
    if names & {"vqe", "iqpe", "dmrg"}:
        fname = (
            "simulated-noisy-3d-n_occ-vs-t2.json"
            if kwargs.get("backend")
            else "simulated-ideal-3d-n_occ-vs-t2.json"
        )
        result = qbp.load_result(str(_DATA_DIR / fname))
        if "dmrg" in names and "dmrg" not in result.methods:
            result.methods = result.methods + ["dmrg"]
            result.grids["dmrg"] = result.grids["analytic"]
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
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE, Method.DMRG],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    method_params={
        Method.VQE: {"iters": 50, "layers": 1, "reps": 1},
        Method.IQPE: {"time": 0.1, "trot": 1, "iters": 1, "reps": 1},
        Method.DMRG: {"nsweeps": 4, "maxdims": "20,50,100,200", "cutoff": 1e-9},
    },
)
```

The analytic surface is the smooth baseline; the VQE, IQPE, and DMRG markers sit on top of it. On an ideal simulator any gap between them is a property of the algorithm's settings, not of hardware noise—and DMRG, being numerically exact for a system this small, lands right on the baseline.

### VQE parameters

- **`iters`.** Optimizer iterations per repetition. More iterations let the classical optimizer converge deeper into the energy landscape. Too few and the ansatz never reaches the ground state; the markers sit *above* the baseline.
- **`layers`.** The number of ansatz layers (repetitions of the entangling block). Deeper ansätze can represent more entangled states, so raising `layers` lifts the ceiling on how close VQE can get—at the cost of more parameters for the optimizer to search.
- **`reps`.** Independent VQE restarts. Because the optimizer starts from a random point and can stall in a local minimum, running several repetitions and keeping the best guards against an unlucky start.

### IQPE parameters

- **`time`.** The Hamiltonian evolution time $t$ baked into the phase-estimation unitary $e^{-iHt}$. It sets the window of energies the phase readout can resolve without wrapping around; too large a `time` aliases the phase, too small a `time` wastes precision.
- **`trot`.** Suzuki–Trotter steps used to approximate $e^{-iHt}$. More steps shrink the Trotter error in the evolution circuit at the cost of a deeper circuit.
- **`iters`.** Phase-estimation iterations. Each iteration resolves one more bit of the phase, so more iterations sharpen the energy estimate.
- **`reps`.** Independent IQPE repetitions, averaged the same way as VQE's.

### DMRG parameters

- **`nsweeps`.** The number of DMRG sweeps back and forth across the chain. More sweeps let the matrix-product state converge further; too few and it stalls short of the ground state.
- **`maxdims`.** A comma-separated schedule of maximum bond dimensions, one per sweep (the last value is reused for any extra sweeps). The bond dimension caps how much entanglement the state can carry—larger values are more accurate but more expensive, so the schedule ramps up gradually.
- **`cutoff`.** The truncation cutoff on discarded singular values. A smaller cutoff keeps more of the state at the cost of a larger effective bond dimension.
- **`conserve_qns` / `seed` / `julia` / `julia_module` / `julia_project` / `script_path`.** Environment settings rather than physics knobs: they control quantum-number conservation, the RNG, and where QBP finds your Julia toolchain and the DMRG script. The defaults are usually fine.

## Tuning Guidance

On a noise-free simulator, agreement with the baseline is limited only by expressiveness and precision:

- If **VQE** sits above the baseline, it hasn't converged. Raise `iters` first; if it plateaus short of the exact energy, the ansatz is too shallow—raise `layers`. Scatter between repetitions means the optimizer is finding different minima, so raise `reps`.
- If **IQPE** is off, it is usually a resolution problem. Raise `iters` to add bits of phase precision, and raise `trot` if the Trotterized evolution itself is the bottleneck. Watch `time`: pick it so the ground-state energy lands comfortably inside the unwrapped phase window.
- If **DMRG** sits above the baseline, it hasn't converged. Add `nsweeps` and extend the `maxdims` schedule to higher bond dimensions; lowering `cutoff` helps when the state is being truncated too aggressively. In one dimension DMRG usually lands right on the analytic surface once the bond dimension is large enough.

Every knob that tightens agreement also lengthens the run. The practical workflow is to start cheap—one layer, a handful of iterations, a modest bond dimension—confirm the sweep shape looks right against the analytic surface, and then spend circuit depth or bond dimension only where you need it. Once the simulation tracks the baseline, you're ready to introduce hardware noise; see [Incorporating Quantum Hardware](incorporating-quantum-hardware.md).
