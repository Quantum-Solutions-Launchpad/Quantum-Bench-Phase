"""Dynamical decoupling

Applies an XY4 pulse sequence during circuit idle periods via a transpiler
scheduling pass to suppress T1/T2 decoherence
"""

from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
from qiskit.circuit.library import XGate, YGate

from qbp._mitigation import MitigationStrategy

# Fallback single/two-qubit gate durations used when the backend has no calibrated timing data to schedule
# against (look DDStrategy.transform_circuit). Each tuple needs an explicit
# "s" unit: InstructionDurations defaults an unmarked numeric duration to
# "dt" (device clock ticks), not seconds, so these values would otherwise be
# misinterpreted as about 9 orders of magnitude too large for our current setup.
_DEFAULT_GATE_DURATIONS = [
    ("id", None, 50e-9, "s"), ("sx", None, 50e-9, "s"), ("x", None, 50e-9, "s"),
    ("y", None, 50e-9, "s"), ("rz", None, 0.0, "s"), ("cx", None, 300e-9, "s"),
    ("ecr", None, 300e-9, "s"), ("cz", None, 300e-9, "s"),
    ("reset", None, 1e-6, "s"), ("measure", None, 1e-6, "s"),
]
_DEFAULT_DT = 1e-9  # 1 ns clock resolution, only used to bridge s -> dt internally


class DDStrategy(MitigationStrategy):
    """XY4 dynamical decoupling via a transpiler scheduling pass

    InstructionDurations.from_backend(AerSimulator(...)) returns an EMPTY
    duration table. AerSimulator (unlike a real or fake backend)
    has no calibrated gate timings which makes ALAPScheduleAnalysis raise
    TranspilerError immediately if used as is. Falls back to a plausible
    default duration table so DD actually runs.
    """

    name = "dd"

    def transform_circuit(self, circuit, backend):
        if backend is None:
            return circuit
        try:
            durations = InstructionDurations.from_backend(backend)
            try:
                durations.get("sx", 0)
            except Exception:
                durations = InstructionDurations(_DEFAULT_GATE_DURATIONS, dt=_DEFAULT_DT)
            xy4 = [XGate(), YGate(), XGate(), YGate()]
            pm = PassManager([
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(durations, xy4),
            ])
            return pm.run(circuit)
        except Exception:
            return circuit
