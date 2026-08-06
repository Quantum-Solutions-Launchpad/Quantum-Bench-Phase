# Built-in Strategies: ZNE, DD, and M3

QBP ships three built-in mitigation strategies. ZNE and DD both work by
rewriting the circuit before it runs (the `transform_circuit` hook from the
[strategy architecture](overview.md#strategy-architecture)); M3 instead
post-processes the measured counts afterward (`correct_counts`). All three
target a different physical noise source, and as the DD benchmarks below
show, are not interchangeable.

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

## Zero-Noise Extrapolation

Gate error accumulates with circuit depth. ZNE deliberately amplifies it at
several controlled scale factors, then extrapolates the resulting
observable back to the zero-noise limit. The amplification is done by
**unitary folding**: replacing a circuit $C$ with

$$
C \;\rightarrow\; C \, (C^{-1} C)^n \qquad \text{(scale factor } 2n+1\text{)}
$$

which leaves the noiseless action of the circuit unchanged (the extra
$C^{-1}C$ pairs cancel exactly on paper) while multiplying its exposure to
physical gate error. QBP fits a straight line through several scale
factors and reads off the zero-noise intercept.

Because it operates on an expectation value rather than raw counts, ZNE
applies to `Method.VQE`:

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

shared = dict(
    model="haldane", lattice=(2, 2),
    x_param="n_occ", x_range=(2, 6, 2),
    y_param="t2", y_range=(0.2, 0.8, 0.3),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    backend=AerSimulator(noise_model=gate_noise()),
)

raw = qbp.run(**shared, method=[Method.ANALYTIC, Method.VQE],
              method_params={Method.VQE: {"iters": 3000, "layers": 4, "reps": 4}})

zne = qbp.run(**shared, method=[Method.ANALYTIC, Method.VQE],
              method_params={Method.VQE: {"iters": 3000, "layers": 4, "reps": 4,
                                           "mitigation": {"zne": True}}})
```

Against pure gate error, ZNE consistently reduces VQE's mean absolute
error relative to the analytic ground truth. The extrapolation removes
most of the depth-dependent bias that raw execution accumulates. The scale
factors used for folding default to `[1, 3, 5]` and can be overridden with something like 
`{"zne": True, "zne_noise_factors": [1, 3, 5, 7]}`.

## Dynamical Decoupling

Decoherence degrades a qubit's state over time even when nothing is being
done to it,including the idle stretches between gates in a circuit. DD
inserts a refocusing pulse sequence (QBP uses XY4 by default) into those
idle windows, identified by a transpiler-level scheduling pass
(`ALAPScheduleAnalysis` + `PadDynamicalDecoupling`). This is the same
principle as spin-echo sequences in nuclear magnetic resonance: a pulse
sequence that refocuses a noise process which stays roughly constant
across the sequence's timescale.

```{jupyter-execute}
from qiskit_aer.noise import thermal_relaxation_error

def decoherence_noise(t1_us=150.0, t2_us=100.0):
    nm = NoiseModel()
    t1, t2 = t1_us * 1e-6, t2_us * 1e-6
    for gate, gt in [("sx", 50e-9), ("x", 50e-9)]:
        nm.add_all_qubit_quantum_error(thermal_relaxation_error(t1, t2, gt), gate)
    err = thermal_relaxation_error(t1, t2, 300e-9)
    nm.add_all_qubit_quantum_error(err.tensor(err), ["ecr", "cx"])
    return nm

shared_dd = {**shared, "backend": AerSimulator(noise_model=decoherence_noise())}

raw_dd = qbp.run(**shared_dd, method=[Method.ANALYTIC, Method.VQE],
                  method_params={Method.VQE: {"iters": 3000, "layers": 4, "reps": 4}})

dd = qbp.run(**shared_dd, method=[Method.ANALYTIC, Method.VQE],
             method_params={Method.VQE: {"iters": 3000, "layers": 4, "reps": 4,
                                          "mitigation": {"dd": True}}})
```

### DD is not free, and not always the right tool

DD only cancels noise that stays roughly *static* across the refocusing
sequence, or something like a slowly drifting dephasing error. It provides no
benefit against **Markovian** (memoryless) decoherence, since there is no
persistent noise process left for the echo to reverse. Also, the pulses it
inserts add real circuit depth, which is itself extra exposure to gate and
decoherence error. In our benchmarks, against the Markovian decoherence
model above, DD was roughly neutral for VQE, but helped more clearly on real hardware. Also importantly, VQE's classical optimizer averages over many noisy measurements, which makes it comparatively tolerant of DD's added depth; IQPE's bit-by-bit classical-feedback decoding is more fragile to it.

```{warning}
Enable DD when you have reason to believe your noise source is
quasi-static (e.g. slow parameter drift), not as a default "more
mitigation is always better" choice. Against Markovian decoherence
specifically, it can make results worse. Verify with a raw-vs-DD
comparison on your own noise model before trusting it in production, the
same way the numbers above were produced.
```

For a matching runnable comparison against exactly this Markovian model,
including the numerical summary table, see `examples/test_iqpe_dd.py` and
`examples/compare_mitigation.py` in the QBP repository.

## M3 Readout Correction

Unlike ZNE and DD, M3 does not touch the circuit at all, it is a purely
classical `correct_counts` post-processing step, applied *after*
measurement, that targets readout error: the final-step misreporting of a
qubit's true state as the opposite outcome.

M3 (matrix-free measurement mitigation) calibrates a per-qubit confusion
matrix, the probability that a qubit prepared in $|0\rangle$ or
$|1\rangle$ is read out as the other state, and inverts it to correct the
measured bitstring distribution. Naively, correcting an $n$-qubit
distribution requires inverting a $2^n \times 2^n$ matrix; M3's
contribution is doing this *without* ever constructing that matrix
explicitly, which is what keeps calibration and correction tractable well
past the qubit counts where the naive approach becomes intractable.

Because it corrects a measured count distribution rather than a computed
expectation value, M3 applies to `Method.IQPE`, whose per-iteration output
*is* a bitstring distribution from the ancilla qubit, rather than
`Method.VQE`, whose measurement is already reduced to a scalar expectation
value before mitigation would have anything to act on. (Pairing M3 with
VQE requires switching VQE's measurement to a counts-based estimator
first (see [Writing a Custom Strategy](custom-strategies.md) if you want
to experiment with that pairing.)

```{note}
M3 depends on the separate [`mthree`](https://github.com/Qiskit-Partners/mthree)
package (`pip install mthree`), not just `qiskit`/`qiskit-aer`. If it
isn't installed, `{"m3": True}` raises `ImportError` at calibration time
with that install instruction, rather than failing silently.
```

```{jupyter-execute}
from qiskit_aer.noise import ReadoutError

def readout_noise(p=0.08):
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm

shared_m3 = {**shared, "backend": AerSimulator(noise_model=readout_noise())}
iqpe_params = {"time": 0.5, "trot": 4, "iters": 6, "reps": 4}

raw_m3 = qbp.run(**shared_m3, method=[Method.ANALYTIC, Method.IQPE],
                  method_params={Method.IQPE: iqpe_params})

m3 = qbp.run(**shared_m3, method=[Method.ANALYTIC, Method.IQPE],
             method_params={Method.IQPE: {**iqpe_params, "mitigation": {"m3": True}}})
```

Against a readout-error-only noise model like the one above, M3 removes
most of the bias readout error introduces into IQPE's bit-decoding, since
that bias is exactly what its confusion-matrix inversion is built to
correct. It is largely orthogonal to gate error and decoherence. So pairing
it with DD (`{"m3": True, "dd": True}`) is reasonable when both readout
and idle-time decoherence are present, since the two corrections act on
different stages of the pipeline (`transform_circuit` before execution,
`correct_counts` after). QBP applies active strategies in a fixed order:
DD, then ZNE, then M3 regardless of the order their keys appear in the
`mitigation` dict.

## Next

[Writing a Custom Strategy](custom-strategies.md), implement a new
mitigation technique against the same `MitigationStrategy` interface these
three use.
