# Writing a Custom Strategy

[ZNE, DD, and M3](mitigation.md) all implement the same small interface:
`qbp._mitigation.MitigationStrategy`. Adding a new technique or
pairing an existing one with a method it isn't wired up for yet, like M3
with VQE, means writing one more subclass and registering it, rather
than modifying `Method.VQE` or `Method.IQPE` themselves.

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

## The `MitigationStrategy` interface

```python
class MitigationStrategy:
    name: str = ""

    def calibrate(self, backend) -> None:
        """One-time setup before any circuits run, e.g. building a
        confusion matrix from calibration circuits."""

    def transform_circuit(self, circuit, backend):
        """Rewrite the circuit before execution. Return the (possibly
        unchanged) circuit."""
        return circuit

    def measure(self, circuit, op, params, next_measure):
        """Wrap how an expectation value is obtained. Call next_measure(...)
        to continue the chain, or intercept it entirely."""
        return next_measure(circuit, op, params)

    def correct_counts(self, raw_dist: dict, qubits: list, n_clbits: int) -> dict:
        """Classically post-process a measured bitstring distribution.
        Return the (possibly unchanged) distribution."""
        return raw_dist
```

A subclass only overrides the hooks its technique actually needs. The
base class's defaults handles the rest. `calibrate` runs once before
the run starts; `transform_circuit` and `measure` run per circuit;
`correct_counts` runs once per measured distribution.

## Worked example: a count-threshold denoiser

A simple, self-contained technique: drop any measured outcome below a
probability threshold and renormalize the rest. This is a
`correct_counts` only strategy, structurally the same shape as M3 but
without a calibration step:

```{jupyter-execute}
from qbp._mitigation import MitigationStrategy

class ThresholdStrategy(MitigationStrategy):
    name = "threshold"

    def __init__(self, min_fraction: float = 0.01):
        self.min_fraction = min_fraction

    def correct_counts(self, raw_dist: dict, qubits: list, n_clbits: int) -> dict:
        total = sum(raw_dist.values())
        if total == 0:
            return raw_dist
        kept = {k: v for k, v in raw_dist.items() if v / total >= self.min_fraction}
        kept_total = sum(kept.values())
        return {k: v * total / kept_total for k, v in kept.items()} if kept_total else raw_dist
```

## Registering it

`MitigationConfig` is what turns a `mitigation` dict like `{"dd": True}`
into a list of `MitigationStrategy` instances. Its `build_strategies()`
method is a fixed, explicit if-chain rather than a dynamic registry, so
adding a technique means adding one flag and one line, per
`qbp/_mitigation.py`'s own module docstring:

```python
# qbp/_mitigation.py

@dataclass
class MitigationConfig:
    m3: bool = False
    dd: bool = False
    zne: bool = False
    threshold: bool = False              # 1. add the flag
    threshold_min_fraction: float = 0.01

    def build_strategies(self) -> list[MitigationStrategy]:
        strategies: list[MitigationStrategy] = []
        if self.dd:
            from qbp._mitigation_dd import DDStrategy
            strategies.append(DDStrategy())
        if self.zne:
            from qbp._mitigation_zne import ZNEStrategy
            strategies.append(ZNEStrategy(scale_factors=self.zne_noise_factors))
        if self.m3:
            from qbp._mitigation_m3 import M3Strategy
            strategies.append(M3Strategy())
        if self.threshold:                # 2. instantiate it
            from qbp._mitigation_threshold import ThresholdStrategy
            strategies.append(ThresholdStrategy(self.threshold_min_fraction))
        return strategies
```

Once registered this way, `{"threshold": True, "threshold_min_fraction": 0.02}`
works as a `mitigation` value exactly like the built-in techniques —
`MitigationConfig.coerce()` validates it against the dataclass's known
fields automatically, so a typo'd key still raises a clear
`ValueError` rather than silently doing nothing.

```{note}
This is a source-level extension point, not a plugin API. `qbp/_mitigation.py`
is intentionally small and unabstracted so that adding a technique is a
two line diff rather than a new registration mechanism to learn. If you
build a technique worth sharing, contribute it back the same way ZNE, DD,
and M3 were added.
```

## Composing with existing strategies

`chain_measure`, `chain_correct_counts`, and `transform_circuit_chain`
(also in `qbp/_mitigation.py`) are what let multiple strategies stack
without knowing about each other. `Method.VQE` and `Method.IQPE` always
call the composed hook, never a specific strategy's hook directly. A new
`correct_counts` only strategy like `ThresholdStrategy` above composes
with M3 for free: enabling `{"m3": True, "threshold": True}` runs M3's
confusion-matrix inversion first, then thresholds the result, in the fixed
DD to ZNE to M3 to (your new technique, appended last) order `build_strategies()`
constructs.

## Pairing M3 with VQE

The one cross pairing called out on the [built-in strategies page](mitigation.md#m3-readout-correction), M3 for VQE, isn't a new strategy at all, just a measurement-path
change: VQE's `measure` hook would need a counts-based path (sample the
circuit, get a bitstring distribution, run it through `chain_correct_counts`,
then reduce to an expectation value) instead of its current
expectation-value estimator. That's a larger change than the pattern
above, since it touches `Method.VQE`'s `measure` hook rather than adding a
self-contained strategy.

## Next

Back to the [strategy architecture overview](overview.md#strategy-architecture),
or [Built-in Strategies](mitigation.md) for how ZNE, DD, and M3 use this
same interface.
