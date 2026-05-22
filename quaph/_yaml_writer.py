from __future__ import annotations

from typing import Any

import yaml

from quaph._yaml_model import YamlModelSpec


class _LiteralStr(str):
    pass


def _str_presenter(dumper, data):
    if isinstance(data, _LiteralStr) or "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter, Dumper=yaml.SafeDumper)
yaml.add_representer(_LiteralStr, _str_presenter, Dumper=yaml.SafeDumper)


def spec_to_yaml(spec: YamlModelSpec) -> str:
    data: dict[str, Any] = spec.model_dump(by_alias=True, exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, width=120, allow_unicode=True)


def write_spec(spec: YamlModelSpec, path) -> None:
    with open(path, "w") as f:
        f.write(spec_to_yaml(spec))
