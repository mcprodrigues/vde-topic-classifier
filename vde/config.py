"""Load the versioned configuration file."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path

from vde.paths import CONFIG_PATH

# Sections that only steer offline evaluation. Every other section changes what publish computes.
OFFLINE_SECTIONS = frozenset({"evaluation", "forecast"})


def load_config(path: Path = CONFIG_PATH, overrides: dict | None = None) -> dict:
    """Return the configuration as a nested dict, optionally merged with overrides."""
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def fingerprint(config: dict) -> str:
    """sha256 of every section that changes the publication, so model.json can pin the config it was trained with."""
    relevant = {k: v for k, v in config.items() if k not in OFFLINE_SECTIONS}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
