"""Analytic exact-diagonalization simulation method and core solvers."""

from __future__ import annotations

from scipy.sparse.linalg import eigsh as _sparse_eigsh

from qbp._core import _fmt_params, logger
from qbp._linalg import eigh as _eigh
from qbp._method import Method, SimulationMethod, register_method


# --------------------------------------------------------------------- solvers
def analytic(model, lattice, n_occ, model_params, observable: str = "E"):
    H = model._build_H_matrix(lattice, **model_params)
    eigvals, eigvecs = _eigh(H)
    obs = model.get_observable(observable)
    result = float(obs.analytic(model, lattice, H, eigvals, eigvecs, n_occ, model_params))
    logger.info(f"Analytic [{observable}] ({_fmt_params(lattice, n_occ, model_params)}) = {result}")
    return result


def analytic_bands(model, k_tuple, model_params, observable: str = "E"):
    H = model.bloch_hamiltonian(*k_tuple, **model_params)
    eigvals, eigvecs = _eigh(H)
    obs = model.get_observable(observable)
    if obs.analytic_bloch is None:
        raise ValueError(
            f"Observable '{observable}' on model '{model.name}' has no analytic_bloch backend."
        )
    result = obs.analytic_bloch(model, k_tuple, H, eigvals, eigvecs, model_params)
    k_str = tuple(round(float(x), 3) for x in k_tuple)
    logger.info(f"Analytic [{observable}] (k={k_str}, {_fmt_params((), 0, model_params).split(', ', 2)[-1]}) = {result}")
    return result


def analytic_operator(hamiltonian, extremum="min", label="", observable="E"):
    M = hamiltonian.to_matrix(sparse=True)
    which = "LA" if extremum == "max" else "SA"
    vals = _sparse_eigsh(M, k=1, which=which, return_eigenvectors=False)
    result = float(vals[0])
    label_str = f" ({label})" if label else ""
    logger.info(f"Analytic [{observable}]{label_str} = {result}")
    return result


# ---------------------------------------------------------------------- method
@register_method
class AnalyticMethod(SimulationMethod):
    METHOD = Method.ANALYTIC
    LABEL = "Analytic"
    PARAM_SPECS = []
    SUPPORTS_REAL_SPACE = True
    SUPPORTS_BAND_STRUCTURE = True
    SUPPORTS_OPERATOR = True
    WANTS_PARALLEL = False

    def compute_cell(self, model, lattice, n_occ, cell_params, observable, *,
                     backend, ctx):
        return {"value": float(analytic(model, lattice, n_occ, cell_params, observable=observable))}

    def compute_bloch_cell(self, model, k_tuple, cell_params, observable, *,
                           backend, ctx):
        bands = analytic_bands(model, k_tuple, cell_params, observable=observable)
        return {"bands": [float(b) for b in bands]}

    def compute_operator_cell(self, op, *, extremum, backend, label, observable: str = "E"):
        return {"value": float(analytic_operator(op, extremum, label=label, observable=observable))}

    def reduce(self, cell, *, extremum="min"):
        if "bands" in cell:
            return cell["bands"]
        return float(cell["value"])
