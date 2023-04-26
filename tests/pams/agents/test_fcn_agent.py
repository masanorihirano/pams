import math
import random
from typing import List
from typing import Optional
from typing import cast

import pytest

from pams import Fundamentals
from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import FCNAgent
from pams.agents.fcn_agent import MARGIN_FIXED
from pams.agents.fcn_agent import MARGIN_NORMAL
from pams.logs import Logger
from tests.pams.agents.test_base import TestAgent


class TestFCNAgent(TestAgent):
    def test_is_finite(self) -> None:
        logger = Logger()
        sim = Simulator(
            prng=random.Random(42), logger=logger, fundamental_class=Fundamentals
        )
        agent = FCNAgent(
            agent_id=0,
            prng=random.Random(42),
            simulator=sim,
            name="fcn_agent",
            logger=logger,
        )
        assert not agent.is_finite(float("nan"))
        assert not agent.is_finite(float("inf"))
        assert not agent.is_finite(-float("inf"))
        assert agent.is_finite(1.0)
        assert agent.is_finite(1)

    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        agent = FCNAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": 1.0,
            "chartWeight": 2.0,
            "noiseWeight": 3.0,
            "noiseScale": 0.001,
            "timeWindowSize": 100,
            "orderMargin": 0.1,
            "marginType": "fixed",
            "meanReversionTime": 200,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        assert agent.asset_volumes == {0: 50, 1: 50, 2: 50}
        assert agent.cash_amount == 10000
        assert agent.is_market_accessible(0)
        assert agent.is_market_accessible(1)
        assert agent.is_market_accessible(2)
        assert agent.get_asset_volume(0) == 50
        assert agent.get_asset_volume(1) == 50
        assert agent.get_asset_volume(2) == 50
        assert agent.get_cash_amount() == 10000
        assert agent.fundamental_weight == 1.0
        assert agent.chart_weight == 2.0
        assert agent.noise_weight == 3.0
        assert agent.noise_scale == 0.001
        assert agent.time_window_size == 100
        assert agent.order_margin == 0.1
        assert agent.margin_type == MARGIN_FIXED
        assert agent.mean_reversion_time == 200

        agent = FCNAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings2 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": [0.0, 1.0],
            "chartWeight": [1.0, 2.0],
            "noiseWeight": [2.0, 3.0],
            "noiseScale": [0.0001, 0.001],
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
            "marginType": "normal",
            "meanReversionTime": [200, 300],
        }
        agent.setup(settings=settings2, accessible_markets_ids=[0, 1, 2])
        assert 0.0 <= agent.fundamental_weight < 1.0
        assert 1.0 <= agent.chart_weight < 2.0
        assert 2.0 <= agent.noise_weight < 3.0
        assert 0.0001 <= agent.noise_scale < 0.001
        assert 100 <= agent.time_window_size < 200
        assert 0.0 <= agent.order_margin < 0.1
        assert 200 <= agent.mean_reversion_time < 300
        assert agent.margin_type == MARGIN_NORMAL

        agent = FCNAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings3 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": 1.0,
            "chartWeight": 2.0,
            "noiseWeight": 3.0,
            "noiseScale": 0.001,
            "timeWindowSize": 100,
            "orderMargin": 0.1,
            "marginType": "unknown",
            "meanReversionTime": 200,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings3, accessible_markets_ids=[0, 1, 2])
        agent = FCNAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings4 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": 1.0,
            "chartWeight": 2.0,
            "noiseWeight": 3.0,
            "noiseScale": 0.001,
            "timeWindowSize": 100,
            "orderMargin": 0.1,
            "marginType": "fixed",
        }
        agent.setup(settings=settings4, accessible_markets_ids=[0, 1, 2])
        assert agent.mean_reversion_time == 100

    def test__repr(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        _prng = random.Random(42)
        agent = FCNAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent", logger=logger
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": 1.0,
            "chartWeight": 2.0,
            "noiseWeight": 3.0,
            "noiseScale": 0.001,
            "timeWindowSize": 100,
            "orderMargin": 0.1,
            "marginType": "fixed",
            "meanReversionTime": 200,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        assert (
            str(agent)
            == f"<pams.agents.fcn_agent.FCNAgent | id=1, rnd={_prng}, chart_weight=2.0, fundamental_weight=1.0, "
            f"noise_weight=3.0, is_chart_following:True, margin_type=0, mean_reversion_time:200, noise_scale=0.001, "
            f"time_window_size=100, order_margin=MARGIN_NORMAL"
        )

    @pytest.mark.parametrize("margin_type", [None, "fixed", "normal"])
    @pytest.mark.parametrize("seed", [1, 42, 100, 200])
    def test_submit_orders(self, margin_type: Optional[str], seed: int) -> None:
        sim = Simulator(prng=random.Random(seed + 1))
        logger = Logger()
        _prng = random.Random(seed)
        agent = FCNAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent", logger=logger
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": 1.0,
            "chartWeight": 2.0,
            "noiseWeight": 3.0,
            "noiseScale": 0.001,
            "timeWindowSize": 100,
            "orderMargin": 0.1,
            "marginType": margin_type,
            "meanReversionTime": 200,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        market = Market(
            market_id=0,
            prng=random.Random(seed - 1),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market._update_time(next_fundamental_price=300.0)
        orders = cast(List[Order], agent.submit_orders(markets=[market]))
        order = orders[0]

        _prng = random.Random(seed)
        fundamental_log = 0
        chart_log = 0
        noise_log = 0.001 * _prng.gauss(mu=0.0, sigma=1.0)
        exp_log_return = (fundamental_log + chart_log * 2 + noise_log * 3) / (1 + 2 + 3)
        exp_future_price = 300.0 * math.exp(exp_log_return * 100)
        if exp_future_price > 300:
            if margin_type != "normal":
                price = exp_future_price * 0.9
            else:
                price = exp_future_price + 0.1 * _prng.gauss(mu=0.0, sigma=1.0)
            assert order.is_buy
        else:
            if margin_type != "normal":
                price = exp_future_price * 1.1
            else:
                price = exp_future_price + 0.1 * _prng.gauss(mu=0.0, sigma=1.0)
            assert not order.is_buy
        assert order.price == price
        assert order.ttl == 100

        market = Market(
            market_id=4,
            prng=random.Random(1),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        agent.submit_orders(markets=[market])
