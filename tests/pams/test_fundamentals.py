import random

from pams.fundamentals import Fundamentals


class TestFundamentals:
    def test__(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=0, initial=1000, drift=0.1, volatility=0.2)
        f.add_market(market_id=1, initial=500, drift=0.0, volatility=0.1)
        f.add_market(market_id=2, initial=500, drift=0.0, volatility=0.1, start_at=10)
        f.get_fundamental_price(market_id=0, time=0)
