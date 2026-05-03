from __future__ import annotations

from quaph._model import Model

from quaph.models.haldane import model as _haldane
from quaph.models.hubbard import model as _hubbard
from quaph.models.haldane_hubbard import model as _haldane_hubbard

_MODELS: dict[str, Model] = {
    m.name: m for m in [_haldane, _hubbard, _haldane_hubbard]
}


def get_model(name: str) -> Model:
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODELS)}")
    return _MODELS[name]


def register_model(model: Model) -> None:
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance, got {type(model).__name__}")
    if model.name in _MODELS:
        raise ValueError(f"A model named '{model.name}' is already registered. Use a unique name.")
    _MODELS[model.name] = model
