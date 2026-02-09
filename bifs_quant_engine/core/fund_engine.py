from __future__ import annotations

from typing import Any, Dict

from .utils import ensure_dir
from pathlib import Path


class FundEngine:
    """Very small stub for a fund engine that evaluates strategies and writes a report."""

    def __init__(self, mandate: Dict[str, Any], universe: Dict[str, Any], strategies_cfg: Dict[str, Any]):
        self.mandate = mandate
        self.universe = universe
        self.strategies_cfg = strategies_cfg

    def run_once(self) -> None:
        # For now, just create reports directory and write a minimal file
        pkg_root = Path(__file__).resolve().parents[1]
        out_dir = ensure_dir(pkg_root / "reports")
        report_path = out_dir / "last_run.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("FundEngine run successful\n")
            f.write(f"Mandate keys: {list(self.mandate.keys())}\n")
            f.write(f"Universe keys: {list(self.universe.keys())}\n")
            f.write(f"Strategies: {list(self.strategies_cfg.keys())}\n")
