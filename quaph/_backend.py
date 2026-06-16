"""Backend resolution and the real-vs-simulated execution differences.

QuaPh's ``backend`` argument selects how the quantum methods (VQE/IQPE) execute:

* ``None``              -> ideal :class:`~qiskit_aer.AerSimulator` (no noise),
* a Qiskit **fake** backend (object or name) -> local noisy simulation built
  from that backend's noise model,
* a **real** IBM device (object, device name, or ``"least_busy"``) -> execution
  on hardware via Qiskit Runtime, given the user has credentials configured.

This module owns everything that differs between those cases so the per-method
solvers stay simple: name/string resolution (:func:`resolve_backend`), the
real-backend predicate (:func:`is_real_backend`), and the IQPE sampling
abstraction (:func:`make_iqpe_sampler`) that hides the Aer-V1 vs. Runtime
SamplerV2 result-parsing difference behind a single ``sample_bit`` call.
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


def resolve_backend(backend):
    """Normalize the ``backend`` argument into ``None`` or a Qiskit backend object.

    Accepts ``None``, an already-constructed backend object (fake or real), or a
    string: ``"least_busy"`` for the least-busy real device, ``"Fake*"`` /
    ``"fake_*"`` for a fake backend, or any other name as a real device.
    """
    if backend is None or not isinstance(backend, str):
        return backend
    name = backend
    if name == "least_busy":
        return _service().least_busy(operational=True, simulator=False)
    if name.startswith("Fake") or name.startswith("fake_"):
        return _resolve_fake(name)
    return _service().backend(name)


def is_real_backend(backend) -> bool:
    """True if ``backend`` is a real IBM device (executes via Qiskit Runtime)."""
    try:
        from qiskit_ibm_runtime import IBMBackend
    except Exception:
        return False
    return isinstance(backend, IBMBackend)


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


@contextmanager
def make_iqpe_sampler(backend):
    """Context manager yielding an IQPE sampler with a ``sample_bit(qc)`` method.

    Real devices sample via Runtime ``SamplerV2`` inside a Session; ideal/fake
    backends use the local Aer sampler (preserving existing IQPE numerics).
    """
    if is_real_backend(backend):
        with _RuntimeIQPESampler(backend) as sampler:
            yield sampler
    else:
        yield _AerIQPESampler(backend)
