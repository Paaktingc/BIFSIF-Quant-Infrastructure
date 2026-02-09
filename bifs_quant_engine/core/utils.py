from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return a dict. Returns empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
