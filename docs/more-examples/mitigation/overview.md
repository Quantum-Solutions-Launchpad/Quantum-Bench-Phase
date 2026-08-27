# Error Mitigation

Real and noisy-simulated backends corrupt a circuit's output through gate
error, decoherence, and readout error. **Error mitigation** is a family of
classical and circuit-level techniques that reduce the impact of that noise
on a computed observable *without* the extra physical qubits that full
error correction would require, whcih is aa necessity on today's NISQ-era hardware,
where error-corrected logical qubits are not yet available.

QBP ships three built-in strategies, each targeting a different noise
source:

| Strategy | Targets | Applies to | Enable with |
|---|---|---|---|
| [Zero-noise extrapolation](mitigation.md#zero-noise-extrapolation) (ZNE) | Gate error | `Method.VQE` | `{"zne": True}` |
| [Dynamical decoupling](mitigation.md#dynamical-decoupling) (DD) | Decoherence during idle time | `Method.VQE`, `Method.IQPE` | `{"dd": True}` |
| [M3 readout correction](mitigation.md#m3-readout-correction) | Measurement/readout error | `Method.IQPE` | `{"m3": True}` |

```{warning}
The "Applies to" column matters: `MitigationConfig.coerce()` validates
`mitigation` keys against the known option names (`m3`, `dd`, `zne`), not
against which hooks your chosen method actually calls. Passing `{"zne": True}`
to `Method.IQPE`, or `{"m3": True}` to `Method.VQE`, will not raise an
error, it will run and silently have no effect, since IQPE never invokes
the `measure` hook ZNE relies on, and VQE never invokes the
`correct_counts` hook M3 relies on. Stick to the pairings in the table
above unless [M3-for-VQE](custom-strategies.md#pairing-m3-with-vqe) is on or an
equivalent extension exists.
```

## Enabling a strategy

Every method accepts a `mitigation` dict inside its `method_params` entry.
An empty dict (or omitting the key) means "no mitigation" :

```{jupyter-execute}
:hide-code:

import io
import sys
from pathlib import Path
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()

import qbp
from qbp import Method


def _find_data_dir() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        for candidate in (base / "docs" / "_data", base / "_data"):
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("docs/_data not found relative to cwd")


_DATA_DIR = _find_data_dir()


def _patched_run(*args, **kwargs):
    # Single call in this example, always mitigated: load the pre-computed
    # ZNE-vs-gate-noise result rather than re-running 3000-iteration VQE at
    # doc-build time.
    result = qbp.load_result(str(_DATA_DIR / "sweep-vqe-zne-gate.json"))
    result.plot(hide_plot=kwargs.get("hide_plot", False))
    return result


qbp.run = _patched_run
```

```{jupyter-execute}
import math
import qbp
from qbp import Method
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def gate_noise():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 2), ["ecr", "cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["sx", "x"])
    return nm

result = qbp.run(
    model="haldane-honeycomb",
    lattice=(2, 2),
    method=[Method.ANALYTIC, Method.VQE],
    x_param="n_occ", x_range=(2, 6, 2),
    y_param="t2", y_range=(0.2, 0.8, 0.3),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    method_params={
        Method.VQE: {"iters": 3000, "layers": 4, "reps": 4, "mitigation": {"zne": True}},
    },
    backend=AerSimulator(noise_model=gate_noise()),
)
```

Multiple keys can be combined in the same dict (e.g. `{"dd": True, "m3": True}`)and QBP composes them through the same interface described below,
applying each in a well-defined order rather than requiring the strategies
to know about each other.

## Why mitigation needs a noisy backend

Against a truly noiseless backend, every strategy's *output* is unchanged
from the raw run, but that happens for different reasons depending on the
strategy, and only one of them literally skips its work:

- `backend=None` DD's `transform_circuit` and M3's `calibrate` both
  check for this explicitly and return immediately, doing nothing.
- A bare `AerSimulator()` with no noise model, DD still schedules and
  pads the circuit with real XY4 pulses, and M3 still calibrates (against
  a perfect confusion matrix). Neither is skipped; their *effect* is just
  numerically neutral, since XY4 composes to identity (up to an
  unobservable global phase) with nothing to echo away, and a perfect
  confusion matrix inverts to the identity. ZNE behaves the same way: it
  still folds the circuit and measures at every scale factor, but with no
  noise to amplify, all the scaled measurements come back identical and
  the zero-noise extrapolation returns that same value.

So `mitigation` without real noise will run successfully and match the raw
result but it isn't free: DD and ZNE still do the extra circuit/measurement work in that case, they just don't change the answer.

```{note}
QBP does **not** rely on Qiskit IBM Runtime's built-in resilience options
(`EstimatorOptions.resilience`), because those options are silently inert
for a local `AerSimulator` backend, the common case for the noisy
simulations in QBP. Instead, each strategy below is a
self-contained implementation that runs identically whether the backend is
a local simulator or real hardware.
```

## Strategy architecture

Internally, a strategy is a small object exposing up to four optional
hooks, only some of which a given technique needs:

- `calibrate(backend)` - one-time setup before any circuits run (e.g. M3
  building its readout confusion matrix).
- `transform_circuit(circuit, backend)` - rewrites the circuit before
  execution (e.g. ZNE's folding, DD's pulse padding).
- `measure(circuit, op, params, next_measure)` - wraps how an expectation
  value is obtained; call `next_measure(...)` to continue the chain.
- `correct_counts(raw_dist, qubits, n_clbits)` - classically post-processes
  a measured bitstring distribution (e.g. M3's confusion-matrix inversion).

Strategies compose through `chain_measure`, `chain_correct_counts`, and
`transform_circuit_chain`, so a method implementation calls one composed
hook instead of branching on which techniques are active. This is what
lets `{"dd": True, "m3": True}` just work without VQE or IQPE needing to
know both are enabled. Active strategies run in a fixed order (DD, then
ZNE, then M3) regardless of key order in the `mitigation` dict. See
[Built-in Strategies](mitigation.md) for the full `MitigationStrategy` and
`MitigationConfig` interface, and [Writing a Custom Strategy](custom-strategies.md)
to add your own.

## Next

- [Built-in Strategies](mitigation.md) — ZNE, DD, and M3 in depth,
  including when DD helps versus hurts.
- [Writing a Custom Strategy](custom-strategies.md) — implement a new
  technique against the same interface.

```{toctree}
:hidden:
:maxdepth: 1

mitigation
custom-strategies
```