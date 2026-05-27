from __future__ import annotations

import functools
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from io import BytesIO

from qiskit.quantum_info import SparsePauliOp

_HDF5_SUFFIXES = (".h5", ".hdf5", ".he5")


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _download(source: str) -> bytes:
    import ssl

    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(source, context=context) as resp:
        return resp.read()


def _extract_hdf5_from_zip(data: bytes, source: str) -> str:
    with zipfile.ZipFile(BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            raise ValueError(f"ZIP archive '{source}' contains no files.")
        h5_names = [n for n in names if n.lower().endswith(_HDF5_SUFFIXES)]
        chosen = (h5_names or names)[0]
        dest_dir = tempfile.mkdtemp(prefix="quaph_hamlib_")
        dest = os.path.join(dest_dir, os.path.basename(chosen) or "operator.h5")
        with z.open(chosen) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    return dest


@functools.lru_cache(maxsize=None)
def resolve_hamlib_source(source: str) -> str:
    if _is_url(source):
        data = _download(source)
        if source.lower().endswith(".zip") or zipfile.is_zipfile(BytesIO(data)):
            return _extract_hdf5_from_zip(data, source)
        dest_dir = tempfile.mkdtemp(prefix="quaph_hamlib_")
        dest = os.path.join(dest_dir, os.path.basename(source) or "operator.h5")
        with open(dest, "wb") as out:
            out.write(data)
        return dest

    if zipfile.is_zipfile(source):
        with open(source, "rb") as f:
            return _extract_hdf5_from_zip(f.read(), source)

    return source


def list_hamlib_keys(path: str) -> list[str]:
    import h5py

    keys: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    with h5py.File(resolve_hamlib_source(path), "r") as f:
        f.visititems(visit)

    return sorted(keys)


def _parse_qubit_operator_string(op_string: str) -> SparsePauliOp:
    parsed = []
    max_index = -1
    for coeff_text, ops_text in re.findall(r"([^\[]*)\[([^\]]*)\]", op_string):
        token = "".join(coeff_text.split())
        if token.startswith("+"):
            token = token[1:]
        if not token:
            continue
        coeff = complex(token)
        factors = []
        for factor in ops_text.split():
            pauli, idx = factor[0], int(factor[1:])
            factors.append((pauli, idx))
            max_index = max(max_index, idx)
        parsed.append((coeff, factors))

    n_qubits = max(max_index + 1, 1)
    labels: list[str] = []
    coeffs: list[complex] = []
    for coeff, factors in parsed:
        chars = ["I"] * n_qubits
        for pauli, idx in factors:
            chars[idx] = pauli
        labels.append("".join(reversed(chars)))
        coeffs.append(coeff)

    if not labels:
        labels = ["I" * n_qubits]
        coeffs = [0.0]

    return SparsePauliOp(labels, coeffs).simplify()


def load_hamlib_operator(path: str, key: str) -> SparsePauliOp:
    import h5py

    with h5py.File(resolve_hamlib_source(path), "r", libver="latest") as f:
        op_string = f[key][()].decode("utf-8")
    return _parse_qubit_operator_string(op_string)


_NUMBER = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
_TOKEN_RE = re.compile(rf"(?:^|[_-])([A-Za-z][A-Za-z0-9]*)-({_NUMBER})(?=$|[_-])")


def parse_key_param(key: str, param: str) -> float | None:
    m = re.search(rf"(?:^|[_-]){re.escape(param)}-({_NUMBER})(?=$|[_-])", key)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_key_params(key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, val in _TOKEN_RE.findall(key):
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out
