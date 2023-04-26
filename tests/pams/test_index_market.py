import random

import pytest

from pams import IndexMarket
from pams import Market
from pams import Simulator


class TestIndexMarket:
    def test_init__(self) -> None:
        sim = Simulator(prng=random.Random(32))
        im = IndexMarket(market_id=1, prng=random.Random(42), simulator=sim, name="im")
        assert im._components == []
        assert im.get_components() == []

    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(32))
        m1 = Market(market_id=0, prng=random.Random(42), simulator=sim, name="market")
        m1.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        sim._add_market(market=m1)
        setting = {
            "tickSize": 0.001,
            "outstandingShares": 100,
            "marketPrice": 300.0,
            "fundamentalPrice": 500.0,
            "markets": ["market"],
        }
        im = IndexMarket(market_id=1, prng=random.Random(32), simulator=sim, name="im")
        im.setup(settings=setting)
        setting = {
            "tickSize": 0.001,
            "outstandingShares": 100,
            "marketPrice": 300.0,
            "fundamentalPrice": 500.0,
        }
        im = IndexMarket(market_id=1, prng=random.Random(32), simulator=sim, name="im")
        with pytest.raises(ValueError):
            im.setup(settings=setting)
        setting = {
            "tickSize": 0.001,
            "outstandingShares": 100,
            "marketPrice": 300.0,
            "fundamentalPrice": 500.0,
            "markets": ["market"],
            "requires": ["market"],
        }
        im = IndexMarket(market_id=1, prng=random.Random(32), simulator=sim, name="im")
        with pytest.warns(Warning):
            im.setup(settings=setting)

    def test_add_market(self) -> None:
        sim = Simulator(prng=random.Random(32))
        im = IndexMarket(market_id=1, prng=random.Random(42), simulator=sim, name="im")
        m1 = Market(market_id=0, prng=random.Random(42), simulator=sim, name="market")
        m1.setup(
            settings={
                "tickSize": 0.001,
                # "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        with pytest.raises(AssertionError):
            im._add_market(market=m1)
        m1.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        im._add_market(market=m1)
        with pytest.raises(ValueError):
            im._add_market(market=m1)
        with pytest.raises(ValueError):
            im._add_markets(markets=[m1])

    def test_compute_fundamental_index(self) -> None:
        sim = Simulator(prng=random.Random(32))
        im = IndexMarket(market_id=2, prng=random.Random(42), simulator=sim, name="im")
        m1 = Market(market_id=0, prng=random.Random(42), simulator=sim, name="market1")
        m1.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        im._add_market(market=m1)
        m2 = Market(market_id=1, prng=random.Random(42), simulator=sim, name="market2")
        m2.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 300,
                "marketPrice": 400.0,
                "fundamentalPrice": 400.0,
            }
        )
        im._add_market(market=m2)
        m1._update_time(next_fundamental_price=500.0)
        m2._update_time(next_fundamental_price=400.0)
        im._update_time(next_fundamental_price=450.0)
        assert im.compute_fundamental_index() == 425.0
        assert im.get_fundamental_index() == 450.0
        assert im.get_fundamental_price() == 450.0

    def test_compute_market_index(self) -> None:
        sim = Simulator(prng=random.Random(32))
        im = IndexMarket(market_id=2, prng=random.Random(42), simulator=sim, name="im")
        m1 = Market(market_id=0, prng=random.Random(42), simulator=sim, name="market1")
        m1.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        im._add_market(market=m1)
        m2 = Market(market_id=1, prng=random.Random(42), simulator=sim, name="market2")
        m2.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 300,
                "marketPrice": 400.0,
                "fundamentalPrice": 400.0,
            }
        )
        im._add_market(market=m2)
        m1._update_time(next_fundamental_price=500.0)
        m2._update_time(next_fundamental_price=400.0)
        im._update_time(next_fundamental_price=450.0)
        assert im.compute_market_index() == 375.0
        assert im.get_market_index() == 375.0
        assert im.get_index() == 375.0

    def test_is_all_markets_running(self) -> None:
        sim = Simulator(prng=random.Random(32))
        im = IndexMarket(market_id=2, prng=random.Random(42), simulator=sim, name="im")
        m1 = Market(market_id=0, prng=random.Random(42), simulator=sim, name="market1")
        m1.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        im._add_market(market=m1)
        m2 = Market(market_id=1, prng=random.Random(42), simulator=sim, name="market2")
        m2.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 300,
                "marketPrice": 400.0,
                "fundamentalPrice": 400.0,
            }
        )
        im._add_market(market=m2)
        assert not im.is_all_markets_running()
        m1._is_running = True
        assert not im.is_all_markets_running()
        m2._is_running = True
        assert im.is_all_markets_running()
