"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.errors import InferenceError


def load_yaml_config(path: Path) -> dict:
    """Load a YAML config file and return an empty dict for empty files."""
    try:
        with path.expanduser().open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise InferenceError(f"Config could not be loaded: {path}") from exc
    except yaml.YAMLError as exc:
        raise InferenceError(f"Config is not valid YAML: {path}") from exc

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise InferenceError(f"Config must contain a YAML mapping: {path}")
    return config
