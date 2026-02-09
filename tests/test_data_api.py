import pytest
from pathlib import Path

from bifs_quant_engine.core.data_api import MarketDataAPI


def test_get_prices_for_universe_local_cache():
    api = MarketDataAPI(data_dir=Path("bifs_quant_engine/data"))
    df = api.get_prices_for_universe(["SPY"], start="2000-01-01", end=None)
    assert "SPY" in df.columns
    assert not df.empty
    assert df.index.is_monotonic_increasing
