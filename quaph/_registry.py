from __future__ import annotations

import shutil
import sys
from pathlib import Path

from quaph._model import Model
from quaph._yaml_model import load_yaml_model

_BUILTIN_NAMES = frozenset({
    "haldane", "hubbard", "haldane-hubbard",
    "ssh", "kane-mele", "kane-mele-lc"
})

_MODELS: dict[str, Model] = {}


def _models_dir() -> Path:
    return Path(__file__).parent / "models"


def _module_name_for(model_name: str) -> str:
    return model_name.replace("-", "_")


def _yaml_path_for(model_name: str) -> Path:
    return _models_dir() / f"{_module_name_for(model_name)}.yaml"


def _discover_models() -> None:
    for path in sorted(_models_dir().glob("*.yaml")) + sorted(_models_dir().glob("*.yml")):
        try:
            m = load_yaml_model(path)
        except Exception as e:
            print(f"warning: failed to load model file {path.name}: {e}", file=sys.stderr)
            continue
        _MODELS[m.name] = m


def get_model(name: str) -> Model:
    """Fetch a registered :class:`Model` by name.

    .. todo:: Describe the discovery flow (built-in YAMLs are auto-loaded at
       import time) and link to the model catalog.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"ssh"`` or ``"haldane"``.

    Returns
    -------
    Model
        The registered model.

    Raises
    ------
    ValueError
        If no model with this name is registered.
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODELS)}")
    return _MODELS[name]


def register_model(model: Model) -> None:
    """Register an in-memory :class:`Model` so it can be looked up by name.

    .. todo:: Describe the in-memory vs. on-disk distinction and when to use
       :func:`register_model_from_file` instead.

    Parameters
    ----------
    model : Model
        A constructed :class:`Model` instance.

    Raises
    ------
    TypeError
        If ``model`` is not a :class:`Model`.
    ValueError
        If a model with the same name is already registered.
    """
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance, got {type(model).__name__}")
    if model.name in _MODELS:
        raise ValueError(f"A model named '{model.name}' is already registered. Use a unique name.")
    _MODELS[model.name] = model


def register_model_from_file(path: str | Path) -> Model:
    """Load a YAML model from disk, register it, and persist the file.

    .. todo:: Describe the YAML schema (link to ``models/custom-yaml.md``)
       and explain where the file is copied to under ``quaph/models/``.

    Parameters
    ----------
    path : str or Path
        Path to a ``.yaml``/``.yml`` model spec.

    Returns
    -------
    Model
        The loaded and registered model.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is not ``.yaml``/``.yml``, if a model with the
        same name is already registered, or if the destination file already
        exists in ``quaph/models/``.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"No such file: {src}")
    if src.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"Model file must have .yaml or .yml extension: {src}")
    model = load_yaml_model(src)
    if model.name in _MODELS:
        raise ValueError(f"A model named '{model.name}' is already registered.")
    dst = _yaml_path_for(model.name)
    if dst.exists():
        raise ValueError(f"A model file already exists at {dst}. Remove it first.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    _MODELS[model.name] = model
    return model


def remove_model(name: str) -> None:
    """Unregister a custom model and delete its YAML file if present.

    .. todo:: Describe why built-ins are protected and how to confirm
       removal.

    Parameters
    ----------
    name : str
        Registry key of the model to remove.

    Raises
    ------
    ValueError
        If ``name`` refers to a built-in model, or if no model with this
        name is registered (neither in memory nor on disk).
    """
    if name in _BUILTIN_NAMES:
        raise ValueError(f"Cannot remove built-in model '{name}'.")
    existed_in_memory = _MODELS.pop(name, None) is not None
    path = _yaml_path_for(name)
    existed_on_disk = path.exists()
    if existed_on_disk:
        path.unlink()
    if not existed_in_memory and not existed_on_disk:
        raise ValueError(f"No registered model named '{name}'.")
    print(f"Removed model '{name}'.")


_discover_models()
