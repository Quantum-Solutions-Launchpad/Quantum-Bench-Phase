from __future__ import annotations

import importlib
import inspect
import sys
import textwrap
from pathlib import Path

import quaph.models as _models_pkg
from quaph._model import Model

_BUILTIN_NAMES = frozenset({"haldane", "hubbard", "haldane-hubbard"})

_MODELS: dict[str, Model] = {}


def _models_dir() -> Path:
    return Path(_models_pkg.__file__).parent


def _module_name_for(model_name: str) -> str:
    return model_name.replace("-", "_")


def _file_for(model_name: str) -> Path:
    return _models_dir() / f"{_module_name_for(model_name)}.py"


def _discover_models() -> None:
    for path in sorted(_models_dir().glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        modname = f"quaph.models.{path.stem}"
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            print(f"warning: failed to load model file {path.name}: {e}", file=sys.stderr)
            continue
        m = getattr(mod, "model", None)
        if isinstance(m, Model):
            _MODELS[m.name] = m


def get_model(name: str) -> Model:
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODELS)}")
    return _MODELS[name]


def register_model(
    model: Model,
    *,
    _source_blocks: dict[str, str] | None = None,
) -> None:
    if not isinstance(model, Model):
        raise TypeError(f"Expected a Model instance, got {type(model).__name__}")
    if model.name in _MODELS:
        raise ValueError(f"A model named '{model.name}' is already registered. Use a unique name.")
    path = _file_for(model.name)
    if path.exists():
        raise ValueError(f"A model file already exists at {path}. Remove it first.")
    _write_model_file(path, model, _source_blocks or {})
    _MODELS[model.name] = model


def remove_model(name: str) -> None:
    if name in _BUILTIN_NAMES:
        raise ValueError(f"Cannot remove built-in model '{name}'.")
    existed_in_memory = _MODELS.pop(name, None) is not None
    path = _file_for(name)
    existed_on_disk = path.exists()
    if existed_on_disk:
        path.unlink()
    if not existed_in_memory and not existed_on_disk:
        raise ValueError(f"No registered model named '{name}'.")
    print(f"Removed model '{name}'.")


_CALLABLE_FIELDS = (
    ("hamiltonian_matrix", "_hamiltonian_matrix_fn"),
    ("fermionic_hamiltonian", "_fermionic_hamiltonian_fn"),
    ("get_optimizer", "_get_optimizer_fn"),
    ("mean_field_correction", "_mean_field_correction_fn"),
)


def _write_model_file(path: Path, model: Model, source_blocks: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "from quaph._model import Model",
        "",
    ]

    callable_var_names: dict[str, str] = {}

    for field_name, attr_name in _CALLABLE_FIELDS:
        fn = getattr(model, attr_name)
        if fn is None:
            continue
        if field_name in source_blocks:
            src = source_blocks[field_name].strip("\n")
            lines.append(src)
            lines.append("")
            var = _detect_function_name(src)
            if var is None:
                raise ValueError(
                    f"Could not find a top-level function definition in pasted source for "
                    f"{field_name}."
                )
            callable_var_names[field_name] = var
        else:
            try:
                src = textwrap.dedent(inspect.getsource(fn))
            except (OSError, TypeError) as e:
                raise ValueError(
                    f"Cannot persist model '{model.name}': unable to obtain source for "
                    f"{field_name} ({e})."
                )
            lines.append(src.rstrip())
            lines.append("")
            name = getattr(fn, "__name__", None)
            if not name or name == "<lambda>":
                raise ValueError(
                    f"Cannot persist model '{model.name}': {field_name} has no usable name."
                )
            callable_var_names[field_name] = name

    lines.append("model = Model(")
    lines.append(f"    name={model.name!r},")
    lines.append(f"    display_name={model.display_name!r},")
    lines.append(f"    default_params={model.default_params!r},")
    lines.append(f"    param_labels={model.param_labels!r},")
    for field_name, _ in _CALLABLE_FIELDS:
        if field_name in callable_var_names:
            lines.append(f"    {field_name}={callable_var_names[field_name]},")
    if model.sweep_defaults:
        lines.append(f"    sweep_defaults={model.sweep_defaults!r},")
    lines.append(")")
    lines.append("")

    path.write_text("\n".join(lines))


def _detect_function_name(src: str) -> str | None:
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


_discover_models()
