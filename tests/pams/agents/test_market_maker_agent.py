import random
from typing import List
from typing import cast

import pytest

from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import MarketMakerAgent
from tests.pams.agents.test_base import TestAgent


class TestMarketMakerAgent(TestAgent):
    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(4))
        market1 = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market1"
        )
        settings_market = {
            "tickSize": 0.01,
            "marketPrice": 300.0,
            "outstandingShares": 2000,
        }
        market1.setup(settings=settings_market)
        market1._update_time(next_fundamental_price=300.0)
        sim._add_market(market=market1)
        agent = MarketMakerAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "targetMarket": "market1",
            "netInterestSpread": 0.05,
            "orderTimeLength": 3,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0])
        agent = MarketMakerAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        settings2 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "netInterestSpread": 0.05,
            "orderTimeLength": 3,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings2, accessible_markets_ids=[0])
        agent = MarketMakerAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        settings3 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "targetMarket": 1,
            "netInterestSpread": 0.05,
            "orderTimeLength": 3,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings3, accessible_markets_ids=[0])
        agent = MarketMakerAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        settings4 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "targetMarket": "market1",
            "orderTimeLength": 3,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings4, accessible_markets_ids=[0])

    def test_submit_orders(self) -> None:
        sim = Simulator(prng=random.Random(4))
        market1 = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market1"
        )
        settings_market = {
            "tickSize": 0.01,
            "marketPrice": 300.0,
            "outstandingShares": 2000,
        }
        market1.setup(settings=settings_market)
        market1._update_time(next_fundamental_price=300.0)
        sim._add_market(market=market1)
        agent = MarketMakerAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "targetMarket": "market1",
            "netInterestSpread": 0.05,
            "orderTimeLength": 3,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0])
        orders = cast(List[Order], agent.submit_orders(markets=[market1]))
        for order in orders:
            assert isinstance(order, Order)
        order_buy = list(filter(lambda x: x.is_buy, orders))[0]
        order_sell = list(filter(lambda x: not x.is_buy, orders))[0]
        assert order_buy.price == 300.0 * 0.975
        assert order_sell.price == 300.0 * 1.025
