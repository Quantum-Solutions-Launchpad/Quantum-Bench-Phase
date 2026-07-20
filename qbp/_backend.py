"""Backend resolution and the real-vs-simulated execution differences.

QBP's ``backend`` argument selects how the quantum methods (VQE/IQPE) execute:

* ``None``              -> ideal :class:`~qiskit_aer.AerSimulator` (no noise),
* a Qiskit **fake** backend (object or name) -> local noisy simulation built
  from that backend's noise model,
* a **real** IBM device (object, device name, or ``"least_busy"``) -> execution
  on hardware via Qiskit Runtime, given the user has credentials configured,
* an **IQM Resonance** device (``"iqm_emerald"`` / ``"iqm_garnet"`` /
  ``"iqm_sirius"``) -> execution on hardware via ``iqm-client[qiskit]``, given an
  ``IQM_TOKEN`` is configured.

This module owns everything that differs between those cases so the per-method
solvers stay simple: name/string resolution (:func:`resolve_backend`), the
real-backend predicate (:func:`is_real_backend`), the IQPE sampling abstraction
(:func:`make_iqpe_sampler`), and the VQE estimation abstraction
(:func:`make_vqe_estimator`) that hide the per-provider result-parsing and
execution differences behind uniform ``sample_bit`` / ``estimator`` calls.
"""

from __future__ import annotations

import sys
import time
import warnings
from contextlib import contextmanager

_TRANSIENT_RETRY_ATTEMPTS = 5
_TRANSIENT_RETRY_BASE_DELAY = 1.0

_NO_ACCOUNT_HINT = (
    "No Qiskit Runtime account found. Save one with\n"
    "    from qiskit_ibm_runtime import QiskitRuntimeService\n"
    "    QiskitRuntimeService.save_account(channel='ibm_quantum_platform', "
    "token=..., instance=...)\n"
    "or set the QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE environment variables."
)

_IQM_SERVER_URL = "https://resonance.iqm.tech/"
_IQM_DEVICES = {"iqm_emerald": "emerald", "iqm_garnet": "garnet", "iqm_sirius": "sirius"}
_IQM_NO_TOKEN_HINT = (
    "No IQM Resonance API token found. Generate one on the IQM Resonance dashboard\n"
    "(Dashboard -> Generate token) and export it as:\n"
    "    export IQM_TOKEN=<your token>"
)
_IQM_SHOTS = 1024
_IBM_SAMPLER_SHOTS = 4096
_VQE_ESTIMATOR_SHOTS = 4096


def _service():
    """Connect to Qiskit Runtime using the saved default account / env vars."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        return QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(_NO_ACCOUNT_HINT) from exc


def _resolve_fake(name: str):
    """Resolve a fake backend by class name (``FakeSherbrooke``) or by its
    lowercase ``.name`` (``fake_ibm_brisbane``)."""
    from qiskit_ibm_runtime import fake_provider

    cls = getattr(fake_provider, name, None)
    if cls is not None:
        return cls()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attr in dir(fake_provider):
            if not attr.startswith("Fake"):
                continue
            candidate = getattr(fake_provider, attr)
            try:
                instance = candidate()
            except Exception:
                continue
            if getattr(instance, "name", None) == name:
                return instance
    raise ValueError(f"Unknown fake backend '{name}'.")


def _resolve_iqm(name: str):
    """Resolve an IQM Resonance device (``iqm_emerald`` / ``iqm_garnet`` /
    ``iqm_sirius``) to a Qiskit backend."""
    import os

    try:
        from iqm.qiskit_iqm import IQMProvider
    except ImportError as exc:
        raise RuntimeError("IQM support requires 'pip install qbp[iqm]'.") from exc
    device = _IQM_DEVICES.get(name)
    if device is None:
        raise ValueError(f"Unknown IQM backend '{name}'. Known: {sorted(_IQM_DEVICES)}.")
    token = os.environ.get("IQM_TOKEN")
    if not token:
        try:
            from dotenv import find_dotenv, load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv(find_dotenv(usecwd=True))
            token = os.environ.get("IQM_TOKEN")
    if not token:
        raise RuntimeError(_IQM_NO_TOKEN_HINT)
    provider = IQMProvider(_IQM_SERVER_URL, quantum_computer=device)
    return provider.get_backend()


def resolve_backend(backend):
    """Normalize the ``backend`` argument into ``None`` or a Qiskit backend object.

    Accepts ``None``, an already-constructed backend object (fake or real), or a
    string: ``"least_busy"`` for the least-busy real device, ``"Fake*"`` /
    ``"fake_*"`` for a fake backend, ``"iqm_*"`` for an IQM Resonance device, or
    any other name as a real IBM device.
    """
    if backend is None or not isinstance(backend, str):
        return backend
    name = backend
    if name == "least_busy":
        return _service().least_busy(operational=True, simulator=False)
    if name.startswith("Fake") or name.startswith("fake_"):
        return _resolve_fake(name)
    if name.startswith("iqm_"):
        return _resolve_iqm(name)
    return _service().backend(name)


def _is_ibm_backend(backend) -> bool:
    """True if ``backend`` is a real IBM device (executes via Qiskit Runtime)."""
    try:
        from qiskit_ibm_runtime import IBMBackend
    except Exception:
        return False
    return isinstance(backend, IBMBackend)


def is_iqm_backend(backend) -> bool:
    """True if ``backend`` is an IQM Resonance device (executes via ``.run``)."""
    try:
        from iqm.qiskit_iqm import IQMBackend
    except Exception:
        return False
    return isinstance(backend, IQMBackend)


def is_real_backend(backend) -> bool:
    """True if ``backend`` is remote hardware (IBM or IQM), not a local simulator."""
    return _is_ibm_backend(backend) or is_iqm_backend(backend)


def backend_label(backend) -> str:
    """Result-log label for ``backend``: ``"ideal"``, a device name, or class name."""
    if backend is None:
        return "ideal"
    if is_real_backend(backend):
        return getattr(backend, "name", None) or type(backend).__name__
    return type(backend).__name__


# --------------------------------------------------------------------- sampling
class _AerIQPESampler:
    """IQPE sampler backed by the local Aer V1 ``Sampler`` (ideal / fake noise)."""

    def __init__(self, backend):
        from qbp._core import _make_sampler

        self._sampler = _make_sampler(backend)

    def raw_dist(self, qc) -> dict:
        """The raw quasi-probability distribution, before any bit decision.

        Exposed separately from sample_bit so mitigation strategies (M3)
        can correct it before the majority-vote decision is made. Once
        collapsed to a single bit there's nothing left to correct.
        """
        return self._sampler.run([qc]).result().quasi_dists[0]

    def sample_bit(self, qc) -> int:
        result = self.raw_dist(qc)
        return 1 if result.get(1, 0) > result.get(0, 0) else 0


class _RuntimeIQPESampler:
    """IQPE sampler backed by Qiskit Runtime ``SamplerV2`` on a real device.

    Each IQPE iteration is a separate (adaptive) circuit, so circuits are
    transpiled to the device ISA with a preset pass manager and sampled inside a
    single :class:`~qiskit_ibm_runtime.Session` opened for the whole loop.
    """

    def __init__(self, backend):
        self._backend = backend
        self._session = None
        self._sampler = None
        self._pm = None

    def __enter__(self):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import Session, SamplerV2

        self._session = Session(backend=self._backend)
        self._session.__enter__()
        self._pm = generate_preset_pass_manager(
            optimization_level=3, backend=self._backend
        )
        self._sampler = SamplerV2(mode=self._session)
        return self

    def __exit__(self, *exc):
        return self._session.__exit__(*exc)

    def sample_bit(self, qc) -> int:
        isa_qc = self._pm.run(qc)
        result = self._sampler.run([isa_qc]).result()
        # The IQPE classical register is named "c" in construct_iqpe_circuit.
        counts = result[0].data.c.get_counts()
        ones = sum(n for bit, n in counts.items() if bit.endswith("1"))
        zeros = sum(n for bit, n in counts.items() if bit.endswith("0"))
        return 1 if ones > zeros else 0


class _IqmIQPESampler:
    """IQPE sampler backed by an IQM Resonance device via ``IQMBackend.run``."""

    def __init__(self, backend):
        self._backend = backend

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sample_bit(self, qc) -> int:
        isa_qc = _iqm_transpile(qc, self._backend)
        result = self._backend.run(isa_qc, shots=_IQM_SHOTS).result()
        counts = result.get_counts()
        ones = sum(n for bit, n in counts.items() if bit.replace(" ", "").endswith("1"))
        zeros = sum(n for bit, n in counts.items() if bit.replace(" ", "").endswith("0"))
        return 1 if ones > zeros else 0


@contextmanager
def make_iqpe_sampler(backend):
    """Context manager yielding an IQPE sampler with a ``sample_bit(qc)`` method.

    IQM devices sample via ``IQMBackend.run``; real IBM devices sample via Runtime
    ``SamplerV2`` inside a Session; ideal/fake backends use the local Aer sampler
    (preserving existing IQPE numerics).
    """
    if is_iqm_backend(backend):
        with _IqmIQPESampler(backend) as sampler:
            yield sampler
    elif is_real_backend(backend):
        with _RuntimeIQPESampler(backend) as sampler:
            yield sampler
    else:
        yield _AerIQPESampler(backend)


# ------------------------------------------------------------------- estimation
class _RuntimeVQEEstimator:
    """VQE estimator over Aer (ideal/fake) or a real IBM device via Runtime.

    Real IBM hardware needs qiskit_ibm_runtime's Session-backed Estimator. Ideal/fake backends use
    qiskit_aer.primitives.EstimatorV2 directly instead: it computes exact expectation values (no shot sampling) 
    straight from the noisy density matrix, which is like 45x faster in practice than shot-based sampling, still
    correctly reflects the noise model, and needs no Session at all since there's nothing to queue on remote hardware.
    """

    def __init__(self, backend):
        from qbp._core import _make_simulator

        self._backend = backend
        self._simulator = _make_simulator(backend)
        self._session = None

    def __enter__(self):
        from qiskit import transpile

        self._transpile = lambda c: transpile(
            c, backend=self._simulator, optimization_level=3
        )
        if is_real_backend(self._backend):
            from qiskit_ibm_runtime import Session, Estimator

            self._session = Session(backend=self._simulator)
            self._session.__enter__()
            self.estimator = Estimator(mode=self._session)
        else:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

            self.estimator = AerEstimatorV2.from_backend(self._simulator)
        return self

    def __exit__(self, *exc):
        if self._session is not None:
            return self._session.__exit__(*exc)
        return False

    def transpile(self, circuit):
        return self._transpile(circuit)

def _is_transient_network_error(exc):
    try:
        from requests.exceptions import ConnectionError as ReqConnectionError, SSLError
    except ImportError:
        return False
    return isinstance(exc, (SSLError, ReqConnectionError))


def _run_with_transient_retry(orig_run, args, kwargs):
    for attempt in range(_TRANSIENT_RETRY_ATTEMPTS):
        try:
            return orig_run(*args, **kwargs)
        except Exception as exc:
            if not _is_transient_network_error(exc) or attempt == _TRANSIENT_RETRY_ATTEMPTS - 1:
                raise
            delay = _TRANSIENT_RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"[qbp] transient network error submitting job "
                f"({type(exc).__name__}); retrying in {delay:.0f}s "
                f"(attempt {attempt + 1}/{_TRANSIENT_RETRY_ATTEMPTS - 1})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _drop_unsupported_run_options(backend, *names):
    """Strip run options the backend doesn't accept before they reach its
    ``run``. ``BackendEstimatorV2`` always forwards ``seed_simulator`` (default
    ``None``), which IQM backends don't support and warn about on every job.
    """
    if getattr(backend, "_qbp_run_wrapped", False):
        return backend
    orig_run = backend.run

    def run(*args, **kwargs):
        for name in names:
            kwargs.pop(name, None)
        return _run_with_transient_retry(orig_run, args, kwargs)

    backend.run = run
    backend._qbp_run_wrapped = True
    return backend


def _iqm_transpile(circuit, backend, optimization_level=3):
    """Transpile ``circuit`` to IQM-native gates (``cz``, ``r``).

    ``transpile_to_IQM`` mis-synthesizes some two-qubit blocks (e.g. XXPlusYY /
    Givens rotations used by the VQE warm start) at optimization_level >= 2,
    corrupting the circuit's unitary. Qiskit's own transpiler targeting the same
    native basis is correct, so it is used for direct-CZ devices. Resonator
    devices (non-empty ``computational_resonators``) need ``transpile_to_IQM``'s
    MOVE-gate insertion, which Qiskit cannot do, so they keep the IQM pass.
    """
    arch = getattr(backend, "architecture", None)
    if getattr(arch, "computational_resonators", None):
        from iqm.qiskit_iqm import transpile_to_IQM
        return transpile_to_IQM(circuit, backend=backend, optimization_level=optimization_level)
    from qiskit import transpile
    return transpile(
        circuit, coupling_map=backend.coupling_map,
        basis_gates=["cz", "r"], optimization_level=optimization_level,
    )


class _IqmVQEEstimator:
    """VQE estimator over an IQM Resonance device via ``BackendEstimatorV2``."""

    def __init__(self, backend):
        self._backend = _drop_unsupported_run_options(backend, "seed_simulator")

    def __enter__(self):
        from qiskit.primitives import BackendEstimatorV2

        self.estimator = BackendEstimatorV2(backend=self._backend)
        return self

    def __exit__(self, *exc):
        return False

    def transpile(self, circuit):
        return _iqm_transpile(circuit, self._backend)


@contextmanager
def make_vqe_estimator(backend):
    """Context manager yielding a VQE estimator exposing ``transpile(circuit)`` and
    an ``estimator`` with the PUB-based ``run`` API.

    IQM devices estimate via ``BackendEstimatorV2`` over ``IQMBackend.run``;
    ideal/fake/real-IBM backends use the Aer/Runtime ``Estimator``.
    """
    if is_iqm_backend(backend):
        with _IqmVQEEstimator(backend) as est:
            yield est
    else:
        with _RuntimeVQEEstimator(backend) as est:
            yield est

_IQM_DEFAULT_GATE_SECONDS = {
    "r": 24e-9,
    "prx": 24e-9,
    "cz": 90e-9,
    "move": 70e-9,
    "measure": 400e-9,
    "reset": 350e-6,
    "id": 0.0,
    "delay": 0.0,
    "barrier": 0.0,
}
_IQM_CAL_GATE_MAP = {
    "prx": ("r", "prx"),
    "cz": ("cz",),
    "move": ("move",),
    "measure": ("measure",),
    "reset_wait": ("reset",),
}
_IBM_FALLBACK_LAYER_SECONDS = 5e-7

def _iqm_gate_seconds(backend) -> dict:
    """Map IQM ISA gate name -> duration (seconds) for ``backend``.

    A real ``IQMBackend`` carries per-gate pulse durations in its calibration
    set, which it fetches from the server as metadata (no circuits run, no
    credits spent); the per-locus durations are aggregated to a representative
    mean per gate. Fake backends expose the same timing via ``error_profile``.
    Falls back to representative defaults if neither is available. Cached on the
    backend so the calibration set is fetched only once.
    """
    cached = getattr(backend, "_qbp_gate_seconds", None)
    if cached is not None:
        return cached

    durations = dict(_IQM_DEFAULT_GATE_SECONDS)

    profile = getattr(backend, "error_profile", None)
    if profile is not None:
        for name, ns in (getattr(profile, "single_qubit_gate_durations", None) or {}).items():
            durations[name] = float(ns) * 1e-9
            if name == "prx":
                durations["r"] = float(ns) * 1e-9
        for name, ns in (getattr(profile, "two_qubit_gate_durations", None) or {}).items():
            durations[name] = float(ns) * 1e-9

    client = getattr(backend, "client", None)
    if client is not None:
        try:
            observations = client.get_calibration_set().observations
            aggregated: dict = {}
            for obs in observations:
                field = obs.dut_field
                if field.endswith(".duration"):
                    gate = field.split(".")[1]
                    aggregated.setdefault(gate, []).append(float(obs.value))
            for cal_name, isa_names in _IQM_CAL_GATE_MAP.items():
                values = aggregated.get(cal_name)
                if values:
                    seconds = sum(values) / len(values)
                    for isa_name in isa_names:
                        durations[isa_name] = seconds
        except Exception:
            pass

    try:
        backend._qbp_gate_seconds = durations
    except Exception:
        pass
    return durations


def _critical_path_seconds(circuit, gate_seconds: dict) -> float:
    """ASAP critical-path duration (seconds) of an ISA ``circuit`` given a
    ``{gate_name: seconds}`` map. Gates absent from the map contribute 0."""
    index = {q: i for i, q in enumerate(circuit.qubits)}
    end = [0.0] * circuit.num_qubits
    for instruction in circuit.data:
        qubits = [index[q] for q in instruction.qubits]
        if not qubits:
            continue
        start = max(end[i] for i in qubits)
        finish = start + gate_seconds.get(instruction.operation.name, 0.0)
        for i in qubits:
            end[i] = finish
    return max(end) if end else 0.0


def circuit_qpu_seconds(backend, circuit, shots: int) -> float:
    """Estimated QPU execution seconds for one logical ``circuit`` at ``shots``.

    Computed fully offline (no submission, no credits spent): the circuit is
    transpiled to the device ISA exactly as :func:`run` would, its ASAP-scheduled
    wall-clock duration is measured, and the result is scaled by the shot count.
    IBM uses the calibrated ``Target`` via ``QuantumCircuit.estimate_duration``;
    IQM uses its calibrated (or representative) gate durations over the ISA
    circuit, since the IQM ``Target`` carries no per-gate timing.
    """
    if is_iqm_backend(backend):
        isa = _iqm_transpile(circuit, backend)
        seconds = _critical_path_seconds(isa, _iqm_gate_seconds(backend))
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        try:
            pm = generate_preset_pass_manager(
                optimization_level=3, backend=backend, scheduling_method="asap"
            )
            scheduled = pm.run(circuit)
            seconds = float(scheduled.estimate_duration(backend.target, unit="s"))
        except Exception:
            from qiskit import transpile

            isa = transpile(circuit, backend=backend, optimization_level=3)
            seconds = isa.depth() * _IBM_FALLBACK_LAYER_SECONDS
    return seconds * shots
