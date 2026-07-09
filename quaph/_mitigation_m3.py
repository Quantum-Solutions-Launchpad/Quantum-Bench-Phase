"""M3 matrix-free measurement mitigation (this is good for readout errors)

Calibrates once (in the main process, before parallel dispatch) and applies
the correction to raw measurement distributions. Only the raw
calibration matrices are stored on this object, not the live
mthree.M3Mitigation object itself, so the SimulationMethod instance
can be pickled across joblib worker processes without carrying a live backend connection.
"""

from quaph._mitigation import MitigationStrategy


class M3Strategy(MitigationStrategy):
    name = "m3"

    def __init__(self):
        self._cal_data = None
        self._backend = None

    def calibrate(self, backend) -> None:
        if backend is None:
            return
        try:
            import mthree
        except ImportError:
            raise ImportError(
                "mthree is required for M3 mitigation. Please install it with: pip install mthree"
            )
        from loguru import logger
        logger.info("Calibrating M3 measurement mitigator (runs once)...")
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system()
        self._cal_data = mit.single_qubit_cals
        self._backend = backend
        logger.info("M3 calibration complete.")

    def _mitigator(self):
        """Reconstruct a live M3Mitigation object from stored calibration data."""
        if self._cal_data is None:
            return None
        try:
            import mthree
        except ImportError:
            return None
        mit = mthree.M3Mitigation(self._backend)
        mit.single_qubit_cals = self._cal_data
        return mit

    def correct_counts(self, raw_dist: dict, qubits: list, n_clbits: int) -> dict:
        mit = self._mitigator()
        if mit is None:
            return raw_dist
        counts = {format(k, f"0{n_clbits}b"): max(0.0, v) for k, v in raw_dist.items()}
        try:
            corrected = mit.apply_correction(counts, qubits)
            dist = dict(corrected.nearest_probability_distribution())
            return {int(bs, 2): p for bs, p in dist.items()}
        except Exception:
            return raw_dist
