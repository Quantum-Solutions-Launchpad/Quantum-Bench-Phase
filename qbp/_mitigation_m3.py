"""M3 matrix-free measurement mitigation (this is good for readout errors)

Calibrates once (in the main process, before parallel dispatch) and applies
the correction to raw measurement distributions. Only the raw
calibration matrices are stored on this object, not the live
mthree.M3Mitigation object itself, so the SimulationMethod instance
can be pickled across joblib worker processes without carrying a live backend connection.

IQM Resonance devices (BackendV2) are handled specially. mthree's
``cals_from_system`` calls ``backend.configuration()`` (a BackendV1 API IQM
does not implement) and submits non-native calibration circuits, so on IQM we
build the single-qubit assignment matrices ourselves from IQM-native
calibration circuits. And because the VQE path measures via
``BackendEstimatorV2`` -- which exposes only expectation values, never raw
counts -- M3 cannot hook in through ``correct_counts`` there (that hook is only
reachable from IQPE). For VQE on IQM we therefore run a readout-corrected
estimator subclass that injects the M3 correction on the estimator's own raw
counts, reusing its (correct) basis-rotation, layout and grouping logic.
"""

from qbp._mitigation import MitigationStrategy

_CAL_SHOTS = 20000


def _iqm_single_qubit_cals(backend, shots):
    """Build per-qubit 2x2 readout assignment matrices for an IQM backend.

    Uses two IQM-native calibration circuits (all qubits prepared in |0> and in
    |1>) and reads the marginal single-qubit response. Column ``j`` of each
    matrix is ``[P(measure 0 | prepared j), P(measure 1 | prepared j)]``.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from iqm.qiskit_iqm import transpile_to_IQM

    nq = backend.num_qubits
    qc0 = QuantumCircuit(nq, nq)
    qc0.measure(range(nq), range(nq))
    qc1 = QuantumCircuit(nq, nq)
    qc1.x(range(nq))
    qc1.measure(range(nq), range(nq))
    t0 = transpile_to_IQM(qc0, backend=backend, optimization_level=1)
    t1 = transpile_to_IQM(qc1, backend=backend, optimization_level=1)
    c0 = backend.run(t0, shots=shots).result().get_counts()
    c1 = backend.run(t1, shots=shots).result().get_counts()

    def p_one(counts, q):
        num = tot = 0
        for bitstr, n in counts.items():
            bit = bitstr.replace(" ", "")[::-1][q]
            num += n if bit == "1" else 0
            tot += n
        return num / tot if tot else 0.0

    cals = []
    for q in range(nq):
        a = p_one(c0, q)
        b = p_one(c1, q)
        cals.append(np.array([[1 - a, 1 - b], [a, b]], dtype=np.float32))
    return cals


def _make_m3_estimator(backend, cal_data):
    """A ``BackendEstimatorV2`` that M3-corrects raw counts before computing
    expectation values, reusing the base estimator's preprocessing."""
    import numpy as np
    import mthree
    from qiskit.primitives import BackendEstimatorV2

    class _M3BackendEstimator(BackendEstimatorV2):
        def __init__(self):
            super().__init__(backend=backend)
            self._mit = mthree.M3Mitigation(system=None)
            self._mit.single_qubit_cals = cal_data

        def _calc_expval_map(self, counts, metadata):
            fixed = []
            for count, meta in zip(counts, metadata):
                try:
                    orig = meta["orig_paulis"]
                    n = orig.num_qubits
                    support = list(np.arange(n)[np.logical_or.reduce(orig.z | orig.x, axis=0)])
                    if not support:
                        support = [0]
                    clean = {}
                    for bitstr, freq in count.items():
                        key = bitstr.split(" ", 1)[0]
                        clean[key] = clean.get(key, 0) + freq
                    quasi = self._mit.apply_correction(clean, support)
                    fixed.append(dict(quasi.nearest_probability_distribution()))
                except Exception:
                    fixed.append(count)
            return super()._calc_expval_map(fixed, metadata)

    return _M3BackendEstimator()


class M3Strategy(MitigationStrategy):
    name = "m3"

    def __init__(self):
        self._cal_data = None
        self._backend = None
        self._is_iqm = False
        self._estimator = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_estimator"] = None
        return state

    def calibrate(self, backend) -> None:
        if backend is None:
            return
        try:
            import mthree  # noqa: F401
        except ImportError:
            raise ImportError(
                "mthree is required for M3 mitigation. Please install it with: pip install mthree"
            )
        from loguru import logger
        from qbp._backend import is_iqm_backend

        self._backend = backend
        if is_iqm_backend(backend):
            from qbp._backend import _drop_unsupported_run_options
            _drop_unsupported_run_options(backend, "seed_simulator")
            logger.info("Calibrating M3 readout mitigator on IQM device (runs once)...")
            self._cal_data = _iqm_single_qubit_cals(backend, _CAL_SHOTS)
            self._is_iqm = True
        else:
            import mthree

            logger.info("Calibrating M3 measurement mitigator (runs once)...")
            mit = mthree.M3Mitigation(backend)
            mit.cals_from_system()
            self._cal_data = mit.single_qubit_cals
            self._is_iqm = False
        logger.info("M3 calibration complete.")

    def _mitigator(self):
        """Reconstruct a live M3Mitigation object from stored calibration data."""
        if self._cal_data is None:
            return None
        try:
            import mthree
        except ImportError:
            return None
        mit = mthree.M3Mitigation(system=None) if self._is_iqm else mthree.M3Mitigation(self._backend)
        mit.single_qubit_cals = self._cal_data
        return mit

    def measure(self, circuit, op, params, next_measure):
        """VQE hook. The BackendEstimatorV2 path exposes no raw counts, so on
        IQM we run a readout-corrected estimator instead of the plain one.
        Non-IQM backends fall through to the unmitigated estimator (M3 there is
        only wired for IQPE via ``correct_counts``)."""
        if not self._is_iqm or self._cal_data is None:
            return next_measure(circuit, op, params)
        if self._estimator is None:
            self._estimator = _make_m3_estimator(self._backend, self._cal_data)
        isa_op = op.apply_layout(circuit.layout) if circuit.layout is not None else op
        result = self._estimator.run(pubs=[(circuit, [isa_op], [params])]).result()
        evs = result[0].data.evs
        return float(evs.flat[0]) if hasattr(evs, "flat") else float(evs[0])

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
