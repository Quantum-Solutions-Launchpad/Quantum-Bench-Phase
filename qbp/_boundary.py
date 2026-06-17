"""Boundary-condition handling for qbp real-space runs.

Single source of truth for the ``boundary`` selector and its ``boundary_params``
dict. ``boundary`` chooses ``'periodic'`` or ``'open'`` (the only two accepted
values) and ``boundary_params`` carries the open-boundary geometry/potential
knobs, paired with ``boundary`` the same way ``model_params`` is paired with
``model``. Periodic boundaries take no parameters.
"""

from __future__ import annotations


def _normalize_boundary(boundary: str | None) -> str:
    if boundary is None:
        return "periodic"
    mode = str(boundary).strip().lower()
    if mode not in ("periodic", "open"):
        raise ValueError(
            f"unsupported boundary {boundary!r}; expected 'periodic' or 'open'."
        )
    return mode


def _boundary_mode(params: dict) -> str:
    """Resolve the boundary mode the model layer reads from its params dict."""
    raw = params.get("boundary", params.get("boundary_condition", "periodic"))
    return _normalize_boundary(raw)


def _with_boundary(params: dict | None, boundary: str | None) -> dict:
    """Thread the resolved boundary into the model_params passed to the model.

    Boundary is a dedicated ``run()`` selector, so it must not also appear in
    ``model_params``; the model layer reads the injected ``boundary`` key.
    """
    out = dict(params or {})
    for key in ("boundary", "boundary_condition"):
        if key in out:
            raise ValueError(
                f"set the boundary via boundary=..., not model_params[{key!r}]."
            )
    out["boundary"] = _normalize_boundary(boundary)
    return out


# The open-boundary knobs that live inside ``boundary_params``, mirroring the
# model / model_params split. Periodic boundaries take no parameters.
_OPEN_BOUNDARY_PARAMS = (
    "geometry",
    "radius",
    "center",
    "potential_profile",
    "potential_radius",
    "potential_v0",
    "potential_xi",
)


def _resolve_boundary(boundary: str | None, boundary_params: dict | None) -> tuple[str, dict]:
    """Normalize the boundary selection and its parameter dict.

    Returns the normalized mode and a dict with every ``_OPEN_BOUNDARY_PARAMS``
    key populated (``None`` when unset).
    """
    mode = _normalize_boundary(boundary)
    bp = dict(boundary_params or {})
    if mode == "periodic":
        if bp:
            raise ValueError(
                "boundary_params are only valid for boundary='open'; "
                "periodic boundaries take no parameters."
            )
    else:
        unknown = set(bp) - set(_OPEN_BOUNDARY_PARAMS)
        if unknown:
            raise ValueError(
                f"unknown boundary_params {sorted(unknown)} for boundary='open'; "
                f"allowed: {sorted(_OPEN_BOUNDARY_PARAMS)}."
            )
    return mode, {k: bp.get(k) for k in _OPEN_BOUNDARY_PARAMS}
