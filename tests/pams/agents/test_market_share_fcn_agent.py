import random
from typing import List
from typing import cast

import pytest

from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import MarketShareFCNAgent
from tests.pams.agents.test_base import TestAgent


class TestMarketShareFCNAgent(TestAgent):
    @pytest.mark.parametrize("seed", [1, 42, 100, 200])
    def test_submit_orders(self, seed: int) -> None:
        sim = Simulator(prng=random.Random(seed + 1))
        _prng = random.Random(seed)
        agent = MarketShareFCNAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent"
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
        market1 = Market(
            market_id=0, prng=random.Random(seed - 1), simulator=sim, name="market1"
        )
        market1._update_time(next_fundamental_price=300.0)
        market2 = Market(
            market_id=1, prng=random.Random(seed - 1), simulator=sim, name="market2"
        )
        market2._update_time(next_fundamental_price=300.0)
        market3 = Market(
            market_id=2, prng=random.Random(seed - 1), simulator=sim, name="market3"
        )
        market3._update_time(next_fundamental_price=300.0)
        market_share_test = [0, 0, 0]
        for _ in range(1000):
            orders = cast(
                List[Order], agent.submit_orders(markets=[market1, market2, market3])
            )
            order = orders[0]
            market_share_test[order.market_id] += 1
        assert market_share_test[0] > 300
        assert market_share_test[1] > 300
        assert market_share_test[2] > 300

        agent2 = MarketShareFCNAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent"
        )
        agent2.setup(settings=settings1, accessible_markets_ids=[])
        with pytest.raises(AssertionError):
            agent2.submit_orders(markets=[market1, market2, market3])
