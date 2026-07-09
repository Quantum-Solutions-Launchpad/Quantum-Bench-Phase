"""Backend resolution and the real-vs-simulated execution differences.

QBP's ``backend`` argument selects how the quantum methods (VQE/IQPE) execute:

* ``None``              -> ideal :class:`~qiskit_aer.AerSimulator` (no noise),
* a Qiskit **fake** backend (object or name) -> local noisy simulation built
  from that backend's noise model,
* a **real** IBM device (object, device name, or ``"least_busy"``) -> execution
  on hardware via Qiskit Runtime, given the user has credentials configured,
* an **IQM Resonance** device (``"iqm_emerald"`` / ``"iqm_garnet"`` /
  ``"iqm_sirius"``) -> execution on hardware via ``qiskit-iqm``, given an
  ``IQM_TOKEN`` is configured.

This module owns everything that differs between those cases so the per-method
solvers stay simple: name/string resolution (:func:`resolve_backend`), the
real-backend predicate (:func:`is_real_backend`), the IQPE sampling abstraction
(:func:`make_iqpe_sampler`), and the VQE estimation abstraction
(:func:`make_vqe_estimator`) that hide the per-provider result-parsing and
execution differences behind uniform ``sample_bit`` / ``estimator`` calls.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

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
        raise RuntimeError(_IQM_NO_TOKEN_HINT)
    provider = IQMProvider(_IQM_SERVER_URL, quantum_computer=device, token=token)
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
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import Sampler

        if backend:
            noise_model = NoiseModel.from_backend(backend)
            self._sampler = Sampler(
                backend_options={
                    "noise_model": noise_model,
                    "basis_gates": noise_model.basis_gates,
                }
            )
        else:
            self._sampler = Sampler()

    def sample_bit(self, qc) -> int:
        result = self._sampler.run([qc]).result().quasi_dists[0]
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
        from iqm.qiskit_iqm import transpile_to_IQM

        isa_qc = transpile_to_IQM(qc, backend=self._backend, optimization_level=3)
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

    Preserves the original VQE numerics: circuits are transpiled to the Aer/IBM
    backend and expectation values come from ``qiskit_ibm_runtime`` ``Estimator``
    inside a Session opened for the whole optimization.
    """

    def __init__(self, backend):
        from qbp._core import _make_simulator

        self._simulator = _make_simulator(backend)
        self._session = None

    def __enter__(self):
        from qiskit import transpile
        from qiskit_ibm_runtime import Session, Estimator

        self._session = Session(backend=self._simulator)
        self._session.__enter__()
        self.estimator = Estimator(mode=self._session)
        self._transpile = lambda c: transpile(
            c, backend=self._simulator, optimization_level=3
        )
        return self

    def __exit__(self, *exc):
        return self._session.__exit__(*exc)

    def transpile(self, circuit):
        return self._transpile(circuit)


class _IqmVQEEstimator:
    """VQE estimator over an IQM Resonance device via ``BackendEstimatorV2``."""

    def __init__(self, backend):
        self._backend = backend

    def __enter__(self):
        from qiskit.primitives import BackendEstimatorV2

        self.estimator = BackendEstimatorV2(backend=self._backend)
        return self

    def __exit__(self, *exc):
        return False

    def transpile(self, circuit):
        from iqm.qiskit_iqm import transpile_to_IQM

        return transpile_to_IQM(circuit, backend=self._backend, optimization_level=3)


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
