#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.fund_engine import FundEngine
from core.utils import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the fund engine loop")
    p.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "fund_mandate.yaml"),
                   help="Path to fund mandate YAML")
    p.add_argument("--universe", default=str(PACKAGE_ROOT / "config" / "universe.yaml"),
                   help="Path to universe YAML")
    p.add_argument("--strategies", default=str(PACKAGE_ROOT / "config" / "strategies.yaml"),
                   help="Path to strategies YAML")
    return p.parse_args()


def main():
    args = parse_args()
    mandate = load_yaml(args.config)
    universe = load_yaml(args.universe)
    strategies_cfg = load_yaml(args.strategies)

    engine = FundEngine(mandate=mandate, universe=universe, strategies_cfg=strategies_cfg)
    engine.run_once()


if __name__ == "__main__":
    main()
