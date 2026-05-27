from __future__ import annotations

import fnmatch
import re

from qiskit.quantum_info import SparsePauliOp


def parse_operator_spec(spec: str) -> tuple[str, str | None]:
    if "::" in spec:
        path, pattern = spec.split("::", 1)
        return path, (pattern or None)
    return spec, None


def list_hamlib_keys(path: str, pattern: str | None = None) -> list[str]:
    import h5py

    keys: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    with h5py.File(path, "r") as f:
        f.visititems(visit)

    if pattern is not None:
        keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
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

    with h5py.File(path, "r", libver="latest") as f:
        op_string = f[key][()].decode("utf-8")
    return _parse_qubit_operator_string(op_string)


def parse_key_param(key: str, param: str) -> float | None:
    m = re.search(rf"{re.escape(param)}-([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", key)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
