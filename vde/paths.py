"""Filesystem locations shared by every module."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VDE_ROOT", PACKAGE_DIR.parent))
ANNEX_DIR = Path(os.environ.get("VDE_ANNEX_DIR", ROOT / "anexos"))
ARTIFACT_DIR = Path(os.environ.get("VDE_ARTIFACT_DIR", ROOT / "artifacts"))
OUTPUT_DIR = Path(os.environ.get("VDE_OUTPUT_DIR", ROOT / "outputs"))
CONFIG_PATH = PACKAGE_DIR / "config.toml"
