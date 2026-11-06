from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_config(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuração YAML inválida: {config_path}")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()
    return _merge(_read_config(parent_path), data)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    data = _read_config(config_path)
    data["_config_path"] = str(config_path)
    if "models_file" in data:
        models_path = Path(data["models_file"])
        if not models_path.is_absolute():
            models_path = (config_path.parent / models_path).resolve()
        model_data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
        data["models"] = model_data["models"]
    return data


def resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Caminhos do projeto são resolvidos contra a raiz, não contra o cwd.
    return (ROOT / path).resolve()
