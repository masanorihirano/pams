import random

import pytest

from pams import LIMIT_ORDER
from pams import Market
from pams import Simulator
from pams.agents import TestAgent
from pams.logs import Logger

from .test_base import TestAgent as TestParentAgent


class TestTestAgent(TestParentAgent):
    @pytest.mark.parametrize("seed", [42, 10, 100, 20])
    def test_submit_orders(self, seed: int) -> None:
        sim = Simulator(prng=random.Random(seed + 2))
        logger = Logger()
        agent = TestAgent(
            agent_id=1,
            prng=random.Random(seed),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings1 = {"assetVolume": 50, "cashAmount": 10000}
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        market = Market(
            market_id=0,
            prng=random.Random(seed + 1),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market._update_time(next_fundamental_price=300.0)

        orders = agent.submit_orders(markets=[market])
        _prng = random.Random(seed)
        price = 300 + _prng.random() * 2 * 10 - 10
        volume = _prng.randint(1, 100)
        time_length = _prng.randint(1, 100)
        p = _prng.random()
        if p >= 0.8:
            assert len(orders) == 0
        else:
            assert len(orders) == 1
            assert orders[0].agent_id == 1
            assert orders[0].market_id == 0
            assert orders[0].is_buy == (p < 0.4)
            assert orders[0].kind == LIMIT_ORDER
            assert orders[0].volume == volume
            assert orders[0].price == price
            assert orders[0].ttl == time_length
