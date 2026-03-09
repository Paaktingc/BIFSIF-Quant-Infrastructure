from bifs_quant_engine.core.portfolio import Portfolio, Position


def test_weights_to_orders_simple_buy():
    p = Portfolio(cash=100_000.0)
    prices = {"FOO": 100.0}
    target = {"FOO": 1.0}

    orders = p.weights_to_orders(target, prices)

    # Expect to buy as many whole shares as cash allows: floor(100000/100)=1000
    assert len(orders) == 1
    o = orders[0]
    assert o["symbol"] == "FOO"
    assert o["side"] == "buy"
    assert o["qty"] == 1000


def test_weights_to_orders_with_existing_position_and_sell():
    p = Portfolio(cash=0.0, positions={"FOO": Position("FOO", 50)})
    prices = {"FOO": 100.0}
    # target weight zero -> expect sell what we have
    target = {"FOO": 0.0}

    orders = p.weights_to_orders(target, prices)
    assert len(orders) == 1
    o = orders[0]
    assert o["symbol"] == "FOO"
    assert o["side"] == "sell"
    # sell up to held amount
    assert o["qty"] == 50


# ----- Short Position Tests -----


def test_short_position_via_negative_weight():
    """Target weight of -0.5 with no existing position -> sell_short order."""
    p = Portfolio(cash=100_000.0)
    prices = {"FOO": 100.0}
    target = {"FOO": -0.5}

    orders = p.weights_to_orders(target, prices)
    assert len(orders) == 1
    o = orders[0]
    assert o["symbol"] == "FOO"
    assert o["side"] == "sell_short"
    # -0.5 * 100_000 = 50_000 / 100 = 500 shares
    assert o["qty"] == 500


def test_cover_short_position():
    """Start with short position, target weight 0.0 -> buy (cover) order."""
    p = Portfolio(cash=150_000.0, positions={"FOO": Position("FOO", -500)})
    prices = {"FOO": 100.0}
    target = {"FOO": 0.0}

    orders = p.weights_to_orders(target, prices)
    # Should have a buy order to cover
    buy_orders = [o for o in orders if o["side"] == "buy"]
    assert len(buy_orders) == 1
    assert buy_orders[0]["symbol"] == "FOO"
    assert buy_orders[0]["qty"] == 500


def test_portfolio_value_with_short():
    """Short position should reduce portfolio value when price rises."""
    # Short 100 shares at $50 -> received $5000 in cash
    p = Portfolio(cash=105_000.0, positions={"FOO": Position("FOO", -100)})
    prices = {"FOO": 50.0}
    # value = 105_000 + (-100 * 50) = 105_000 - 5_000 = 100_000
    assert p.value(prices) == 100_000.0

    # If price goes up, short loses value
    prices_up = {"FOO": 60.0}
    # value = 105_000 + (-100 * 60) = 105_000 - 6_000 = 99_000
    assert p.value(prices_up) == 99_000.0


def test_apply_fill_short():
    """apply_fill with sell_short should create negative position and add cash."""
    p = Portfolio(cash=100_000.0)
    p.apply_fill("FOO", 100, "sell_short", 50.0)

    assert "FOO" in p.positions
    assert p.positions["FOO"].quantity == -100
    assert p.cash == 105_000.0  # received 100 * 50 = 5000

