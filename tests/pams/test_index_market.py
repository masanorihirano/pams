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
