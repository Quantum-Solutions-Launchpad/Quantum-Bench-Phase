"""Shared low-level infrastructure for QBP simulation methods.

Technique-specific solver logic lives in the per-method modules
(:mod:`qbp._analytic`, :mod:`qbp._vqe`, :mod:`qbp._iqpe`,
:mod:`qbp._dmrg`). This module holds only what those methods share: logging,
sweep-axis resolution, the noisy/ideal backend constructors, and the
initial-state circuit builders reused across VQE/IQPE and the model layer.
"""

import sys

import numpy as np

from qiskit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler

from loguru import logger

from qbp._backend import is_real_backend


def setup_logging():
    fmt_console_info = "[<bold><green>{time:HH:mm:ss}</green></bold>] <white>{message}</white>"
    fmt_console_debug = "[<dim><white>{time:HH:mm:ss}</white></dim>] <dim>{message}</dim>"

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format=fmt_console_info,
        filter=lambda record: record["level"].name == "INFO",
    )
    logger.add(
        sys.stdout,
        level="DEBUG",
        colorize=True,
        format=fmt_console_debug,
        filter=lambda record: record["level"].name == "DEBUG",
    )

    return logger


def _fmt_params(lattice, n_occ, model_params=None, **extra):
    parts = [f"lattice={tuple(lattice)}", f"n_occ={n_occ}"]
    for k, v in (model_params or {}).items():
        parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _unpack_range(param, range_args, default_step):
    lo, hi = range_args[0], range_args[1]
    st = range_args[2] if len(range_args) > 2 else default_step
    if st is None:
        raise ValueError(
            f"A step is required for sweep axis '{param}' (provide MIN MAX STEP)."
        )
    return lo, hi, st


def resolve_sweep(param: str, range_args, n_orbitals: int, momentum_axes: tuple[str, ...] = ()):
    if param == "n_occ":
        if range_args is None:
            vals = list(range(n_orbitals + 1))
        else:
            lo, hi, st = _unpack_range(param, range_args, 1)
            vals = list(range(int(lo), int(hi) + 1, max(1, int(st))))
        return vals, r"$N_{\text{occ}}$", "n_occ"
    if param in momentum_axes:
        if range_args is None:
            lo, hi, st = -np.pi, np.pi, np.pi / 50
        else:
            lo, hi, st = _unpack_range(param, range_args, np.pi / 50)
        vals = list(np.arange(lo, hi + st / 2, st))
        return vals, param, "momentum"
    if range_args is None:
        raise ValueError(
            f"A sweep range is required when the sweep parameter is '{param}' (not 'n_occ')."
        )
    lo, hi, st = _unpack_range(param, range_args, None)
    vals = list(np.arange(lo, hi + st / 2, st))
    return vals, param, "parameter"


def _resolve_noise_model(backend) -> NoiseModel:
    """Get the NoiseModel for the used backend

    NoiseModel.from_backend() reads calibration data (QubitProperties) off
    a real or fake backend. it does NOT know how to recover the
    noise model already attached to a plain AerSimulator(noise_model=...),
    since that simulator has no calibration data to read 
    so everything that builds its backend as AerSimulator(noise_model=custom_model()
    had its noise silently discarded which is why this now exists.
    """
    if isinstance(backend, AerSimulator):
        return backend.options.noise_model
    return NoiseModel.from_backend(backend)


def _make_simulator(backend):
    if is_real_backend(backend):
        return backend
    if backend:
        noise_model = _resolve_noise_model(backend)
        return AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    return AerSimulator()


def _make_sampler(backend):
    # Do not use run_options={"shots": None}. Exact/no-shot sampling skips
    # measurement and therefore does not apply classical readout errors,
    # which are what M3 is intend to mitigate.
    if backend:
        noise_model = _resolve_noise_model(backend)
        return Sampler(
            backend_options={
                "noise_model": noise_model,
                "basis_gates": noise_model.basis_gates,
            },
        )
    return Sampler()

def _hf_initial_state(n_sites: int, spin: int, n_occ: int, mapper):
    if spin == 2:
        return HartreeFock(n_sites, (n_occ // 2 + n_occ % 2, n_occ // 2), mapper)
    num_modes = n_sites
    label = " ".join(f"+_{i}" for i in range(n_occ))
    bitstr_op = FermionicOp({label: 1.0} if label else {"": 1.0}, num_spin_orbitals=num_modes)
    qubit_op = mapper.map(bitstr_op)
    bits = qubit_op.paulis.x[0]
    qc = QuantumCircuit(len(bits))
    for i, bit in enumerate(bits):
        if bit:
            qc.x(i)
    return qc


def _uniform_initial(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    return qc


def _zero_initial(n_qubits: int) -> QuantumCircuit:
    return QuantumCircuit(n_qubits)