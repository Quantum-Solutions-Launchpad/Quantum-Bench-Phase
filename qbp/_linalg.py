"""Backend-dispatching dense Hermitian eigensolvers for exact diagonalization.

These helpers preserve the semantics of :func:`numpy.linalg.eigh` (a full,
exact diagonalization returning the complete spectrum and eigenvectors) while
opportunistically offloading the work to a GPU when one is available and
otherwise running NumPy's LAPACK path with maximal CPU thread parallelism.

Backend selection is controlled by the ``QBP_EIGH_BACKEND`` environment
variable (``auto``, ``numpy``, ``cupy`` or ``torch``) and GPU offload is only
used for matrices at least ``QBP_GPU_MIN_DIM`` wide so small problems are not
penalised by host/device transfer overhead. GPU backends run in double
precision (``float64``/``complex128``) so results stay bit-for-bit comparable
to the NumPy path.
"""

from __future__ import annotations

import os

import numpy as np

from qbp._core import logger

_BACKEND_ENV = "QBP_EIGH_BACKEND"
_GPU_MIN_DIM_ENV = "QBP_GPU_MIN_DIM"
_DEFAULT_GPU_MIN_DIM = 64

_cupy = False
_torch = False
_logged_backend = None


def _backend_choice():
    return os.environ.get(_BACKEND_ENV, "auto").strip().lower()


def _gpu_min_dim():
    try:
        return max(1, int(os.environ.get(_GPU_MIN_DIM_ENV, _DEFAULT_GPU_MIN_DIM)))
    except (TypeError, ValueError):
        return _DEFAULT_GPU_MIN_DIM


def _get_cupy():
    global _cupy
    if _cupy is False:
        try:
            import cupy as cp

            _cupy = cp if cp.cuda.runtime.getDeviceCount() > 0 else None
        except Exception:
            _cupy = None
    return _cupy


def _get_torch_gpu():
    global _torch
    if _torch is False:
        try:
            import torch

            _torch = torch if torch.cuda.is_available() else None
        except Exception:
            _torch = None
    return _torch


def _to_double(H):
    arr = np.asarray(H)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.float64, copy=False)


def _cupy_eigh(cp, H):
    device = cp.asarray(_to_double(H))
    w, v = cp.linalg.eigh(device)
    return cp.asnumpy(w), cp.asnumpy(v)


def _torch_eigh(torch, H):
    tensor = torch.as_tensor(_to_double(H), device="cuda")
    w, v = torch.linalg.eigh(tensor)
    return w.cpu().numpy(), v.cpu().numpy()


def _numpy_eigh(H):
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=os.cpu_count()):
            return np.linalg.eigh(H)
    except Exception:
        return np.linalg.eigh(H)


def _select_backend(dim):
    choice = _backend_choice()
    if choice == "numpy":
        return "numpy"
    if choice == "cupy":
        return "cupy" if _get_cupy() is not None else "numpy"
    if choice == "torch":
        return "torch" if _get_torch_gpu() is not None else "numpy"
    if dim < _gpu_min_dim():
        return "numpy"
    if _get_cupy() is not None:
        return "cupy"
    if _get_torch_gpu() is not None:
        return "torch"
    return "numpy"


def _announce(backend):
    global _logged_backend
    if backend != _logged_backend:
        _logged_backend = backend
        logger.info(f"Exact diagonalization backend: {backend}")


def eigh(H):
    """Full exact Hermitian diagonalization, GPU-accelerated when available.

    Returns ``(eigvals, eigvecs)`` exactly like :func:`numpy.linalg.eigh`.
    """
    arr = np.asarray(H)
    backend = _select_backend(arr.shape[-1])
    _announce(backend)
    if backend == "cupy":
        cp = _get_cupy()
        try:
            return _cupy_eigh(cp, arr)
        except Exception as exc:
            logger.warning(f"CuPy eigh failed ({exc!r}); falling back to NumPy.")
    elif backend == "torch":
        torch = _get_torch_gpu()
        try:
            return _torch_eigh(torch, arr)
        except Exception as exc:
            logger.warning(f"Torch eigh failed ({exc!r}); falling back to NumPy.")
    return _numpy_eigh(arr)
