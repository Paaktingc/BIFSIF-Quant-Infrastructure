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
