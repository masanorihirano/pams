import contextlib
import random
from typing import Optional

import numpy as np
import pytest

from pams.fundamentals import Fundamentals


class TestFundamentals:
    def test__(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=0, initial=1000, drift=0.1, volatility=0.2)
        f.add_market(market_id=1, initial=500, drift=0.0, volatility=0.1)
        f.add_market(market_id=2, initial=500, drift=0.0, volatility=0.1, start_at=10)
        f.get_fundamental_price(market_id=0, time=0)

    def test__init__(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        rnd = random.Random(42)
        np_random = np.random.default_rng(rnd.randint(0, 2**31))
        assert f._prng.random() == rnd.random()
        assert np.allclose(f._np_prng.random(10), np_random.random(10))

    @pytest.mark.parametrize("market_id", [2, 3])
    @pytest.mark.parametrize("initial", [-100.0, 0.0, 100.0, 200.0])
    @pytest.mark.parametrize("drift", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("volatility", [-0.2, 0.0, 0.2])
    @pytest.mark.parametrize("start_at", [None, 0, 1, 2])
    def test_add_market(
        self,
        market_id: int,
        initial: float,
        drift: float,
        volatility: float,
        start_at: Optional[int],
    ) -> None:
        f = Fundamentals(prng=random.Random(42))
        if start_at is not None:
            with pytest.raises(
                ValueError
            ) if volatility < 0 or initial <= 0.0 else contextlib.nullcontext():
                f.add_market(
                    market_id=market_id,
                    initial=initial,
                    drift=drift,
                    volatility=volatility,
                    start_at=start_at,
                )
        else:
            with pytest.raises(
                ValueError
            ) if volatility < 0 or initial <= 0.0 else contextlib.nullcontext():
                f.add_market(
                    market_id=market_id,
                    initial=initial,
                    drift=drift,
                    volatility=volatility,
                )

    def test_add_market2(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=1, initial=100, drift=-1.0, volatility=1.0)
        with pytest.raises(ValueError):
            f.add_market(market_id=1, initial=100, drift=-1.0, volatility=1.0)

    def test_remove_market(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=1, initial=100, drift=-1.0, volatility=1.0)
        f.get_fundamental_price(market_id=1, time=2)
        f.remove_market(market_id=1)
        assert 1 not in f.market_ids

    def test_change_volatility(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=1, initial=100, drift=-1.0, volatility=1.0)
        assert f.volatilities[1] == 1.0
        f.change_volatility(market_id=1, volatility=0.2)
        assert f.volatilities[1] == 0.2
        with pytest.raises(ValueError):
            f.change_volatility(market_id=1, volatility=-0.2)

    def test_change_drift(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=1, initial=100, drift=-1.0, volatility=1.0)
        assert f.drifts[1] == -1.0
        f.change_drift(market_id=1, drift=0.2)
        assert f.drifts[1] == 0.2

    @pytest.mark.parametrize("corr", [0.1, 0.5, 0.9])
    def test_set_correlation(self, corr: float) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=1, initial=300, drift=-0.01, volatility=0.1)
        f.add_market(market_id=2, initial=100, drift=0.0, volatility=0.2)
        f.add_market(market_id=3, initial=200, drift=0.01, volatility=0.3)
        f.set_correlation(market_id1=1, market_id2=2, corr=corr)
        prices1 = np.asarray(f.get_fundamental_prices(market_id=1, times=range(10000)))
        prices2 = np.asarray(f.get_fundamental_prices(market_id=2, times=range(10000)))
        prices3 = np.asarray(f.get_fundamental_prices(market_id=3, times=range(10000)))
        prices = np.stack([prices1, prices2, prices3])
        returns = np.diff(np.log(prices), axis=-1, n=1)
        coef = np.corrcoef(returns)
        assert abs(coef[1, 0] - corr) < 0.02

        with pytest.raises(ValueError):
            f.set_correlation(market_id1=1, market_id2=2, corr=1.0)
        with pytest.raises(ValueError):
            f.set_correlation(market_id1=1, market_id2=2, corr=-1.0)

    def test_remove_correlation(self) -> None:
        f = Fundamentals(prng=random.Random(42))
        f.add_market(market_id=1, initial=300, drift=-0.01, volatility=0.1)
        f.add_market(market_id=2, initial=100, drift=0.0, volatility=0.2)
        f.add_market(market_id=3, initial=200, drift=0.01, volatility=0.3)
        f.set_correlation(market_id1=2, market_id2=3, corr=0.9)
        prices1 = np.asarray(f.get_fundamental_prices(market_id=1, times=range(10000)))
        prices2 = np.asarray(f.get_fundamental_prices(market_id=2, times=range(10000)))
        prices3 = np.asarray(f.get_fundamental_prices(market_id=3, times=range(10000)))
        prices = np.stack([prices1, prices2, prices3])
        returns = np.diff(np.log(prices), axis=-1, n=1)
        coef = np.corrcoef(returns)
        assert abs(coef[1, 2] - 0.9) < 0.02

        f.remove_correlation(market_id2=2, market_id1=3)
        prices1 = np.asarray(f.get_fundamental_prices(market_id=1, times=range(10000)))
        prices2 = np.asarray(f.get_fundamental_prices(market_id=2, times=range(10000)))
        prices3 = np.asarray(f.get_fundamental_prices(market_id=3, times=range(10000)))
        prices = np.stack([prices1, prices2, prices3])
        returns = np.diff(np.log(prices), axis=-1, n=1)
        coef = np.corrcoef(returns)
        assert abs(coef[1, 2] - 0.0) < 0.02

        f.set_correlation(market_id2=2, market_id1=3, corr=0.9)
        f.set_correlation(market_id1=2, market_id2=3, corr=0.9)
        f.remove_correlation(market_id1=3, market_id2=2)

        with pytest.raises(ValueError):
            f.set_correlation(market_id2=2, market_id1=2, corr=0.9)
        with pytest.raises(ValueError):
            f.remove_correlation(market_id1=2, market_id2=2)
