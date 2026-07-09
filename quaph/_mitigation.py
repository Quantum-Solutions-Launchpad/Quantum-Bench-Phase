"""Error mitigation strategy framework.

Each mitigation technique (M3, DD, ZNE, ...) is implemented as its own
MitigationStrategy subclass in its own module (see _mitigation_m3.py,
_mitigation_dd.py, _mitigation_zne.py for base examples). SimulationMethod subclasses 
(in _vqe.py, _iqpe.py) never import or call a specific technique directly,
they build whichever strategies MitigationConfig has toggled on

To add a new technique: write one new file implementing whichever hooks the new
technique needs, add one new flag to MitigationConfig, add one line to
MitigationConfig.build_strategies().
"""

from __future__ import annotations

from dataclasses import dataclass, field


class MitigationStrategy:
    """Base class for an error mitigation technique.

    Subclasses override only the hooks relevant to their technique and the
    rest keep their no-op defaults. Strategy instances are stored on the owning
    SimulationMethod
    """

    name: str = ""

    def calibrate(self, backend) -> None:

        pass

    def transform_circuit(self, circuit, backend):

        return circuit

    def measure(self, circuit, op, params, next_measure):

        return next_measure(circuit, op, params)

    def correct_counts(self, raw_dist: dict, qubits: list, n_clbits: int) -> dict:

        return raw_dist


def chain_measure(strategies: list[MitigationStrategy], base_measure):
    measure = base_measure
    for strategy in reversed(strategies):
        measure = _wrap_measure(strategy, measure)
    return measure


def _wrap_measure(strategy: MitigationStrategy, inner_measure):
    def wrapped(circuit, op, params):
        return strategy.measure(circuit, op, params, inner_measure)
    return wrapped


def chain_correct_counts(strategies: list[MitigationStrategy], raw_dist: dict, qubits: list, n_clbits: int) -> dict:
    dist = raw_dist
    for strategy in strategies:
        dist = strategy.correct_counts(dist, qubits, n_clbits)
    return dist


def transform_circuit_chain(strategies: list[MitigationStrategy], circuit, backend):
    for strategy in strategies:
        circuit = strategy.transform_circuit(circuit, backend)
    return circuit


# Error mitigation configuration

@dataclass
class MitigationConfig:
    """Base error mitigation configuration shared across all methods

    Each simulation method can subclass this to add method-specific mitigation
    options while inheriting the common ones.  The mitigation key in
    method_params accepts either a dict or a pre-constructed MitigationConfig instance.

    Example
    -------

        quaph.run(
            "haldane",
            method=[Method.VQE, Method.IQPE],
            method_params={
                "vqe": {
                    "iters": 200,
                    "mitigation": {"dd": True, "zne": True},
                },
                "iqpe": {
                    "time": 0.5,
                    "mitigation": {"m3": True},
                },
            },
            backend=FakeSherbrooke(),
        )
    """

    m3: bool = False

    dd: bool = False

    zne: bool = False

    zne_noise_factors: list = field(default_factory=lambda: [1, 3, 5])
    """Noise scale factors for ZNE's unitary folding. Must be odd integers:
    global folding (C -> C(C^-1 C)^n) only has an exact construction for
    scale_factor = 2n+1."""

    @classmethod
    def coerce(cls, value) -> "MitigationConfig":
        """Accept a dict, a MitigationConfig instance, or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            known = {f.name for f in cls.__dataclass_fields__.values()}
            unknown = set(value) - known
            if unknown:
                raise ValueError(
                    f"Unknown mitigation option(s) {sorted(unknown)}; "
                    f"allowed: {sorted(known)}."
                )
            return cls(**value)
        raise TypeError(
            f"mitigation must be a dict or MitigationConfig instance, got {type(value).__name__}"
        )

    def any_active(self) -> bool:
        """Return True if any mitigation technique is enabled."""
        return self.m3 or self.dd or self.zne

    def summary(self) -> dict:
        """JSON-friendly snapshot for logging."""
        return {
            "m3": self.m3,
            "dd": self.dd,
            "zne": self.zne,
            "zne_noise_factors": self.zne_noise_factors,
        }

    def build_strategies(self) -> list[MitigationStrategy]:
        """Instantiate the MitigationStrategy objects for whichever
        techniques are toggled on. Fixed order: DD (circuit-level, applied
        once) before ZNE (wraps each measurement) before M3 (corrects raw
        counts after measurement)
        """
        strategies: list[MitigationStrategy] = []
        if self.dd:
            from quaph._mitigation_dd import DDStrategy
            strategies.append(DDStrategy())
        if self.zne:
            from quaph._mitigation_zne import ZNEStrategy
            strategies.append(ZNEStrategy(scale_factors=self.zne_noise_factors))
        if self.m3:
            from quaph._mitigation_m3 import M3Strategy
            strategies.append(M3Strategy())
        return strategies
