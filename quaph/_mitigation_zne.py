"""Zero-noise extrapolation

qiskit-ibm-runtime's EstimatorOptions.resilience.zne_mitigation is
a fail when the backend is a local AerSimulator rather than a real queued
IBM Runtime service since BackendEstimatorV2 never reads that option for local
backends, so a "ZNE-mitigated" run using it is identical to a raw run. This
implements ZNE manually instead: global unitary folding at several noise
scales, followed by linear (Richardson) extrapolation to the zero-noise
limit.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit

from quaph._mitigation import MitigationStrategy


def _fold_global(qc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Global unitary folding: C -> C (C^-1 C)^n for scale_factor = 2n+1.

    Folding amplifies the circuit's physical noise by scale_factor while
    its ideal action is unchanged (C^-1 C = identity)
    """
    if scale_factor == 1:
        return qc
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(f"ZNE scale_factor must be an odd integer >= 1, got {scale_factor}")
    n_folds = (scale_factor - 1) // 2
    inv = qc.inverse()
    folded = qc.copy()
    for _ in range(n_folds):
        folded = folded.compose(inv).compose(qc)
    return folded


def _zne_extrapolate(scale_factors, values) -> float:
    """Richardson (linear) extrapolation of noise scaled values to the zero noise limit."""
    coeffs = np.polyfit(scale_factors, values, deg=1)
    return float(np.polyval(coeffs, 0.0))


class ZNEStrategy(MitigationStrategy):
    name = "zne"

    def __init__(self, scale_factors=(1, 3, 5)):
        scale_factors = tuple(scale_factors)
        for s in scale_factors:
            if s < 1 or s % 2 == 0:
                raise ValueError(
                    "ZNE scale factors must be odd integers >= 1 (global unitary "
                    f"folding only supports scale=2n+1 exactly); got {s} in {scale_factors}."
                )
        self.scale_factors = scale_factors

    def measure(self, circuit, op, params, next_measure):
        values = [next_measure(_fold_global(circuit, s), op, params) for s in self.scale_factors]
        return _zne_extrapolate(self.scale_factors, values)