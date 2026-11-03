"""Load the versioned configuration file."""

import copy
import tomllib
from pathlib import Path

from vde.paths import CONFIG_PATH


def load_config(path: Path = CONFIG_PATH, overrides: dict | None = None) -> dict:
    """Return the configuration as a nested dict, optionally merged with overrides."""
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
