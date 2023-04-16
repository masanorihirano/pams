import random
from typing import cast

import pytest

from pams import IndexMarket
from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import ArbitrageAgent
from pams.logs import Logger
from tests.pams.agents.test_base import TestAgent


class TestArbitrageAgent(TestAgent):
    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
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
        assert agent.order_volume == 1
        assert agent.order_threshold_price == 0.1
        assert agent.order_time_length == 10

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings2 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1.1,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings2, accessible_markets_ids=[0, 1, 2])

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings3 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings3, accessible_markets_ids=[0, 1, 2])

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings4 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1.1,
            "orderTimeLength": 10,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings4, accessible_markets_ids=[0, 1, 2])

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings5 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1,
            "orderThresholdPrice": 0.1,
        }
        agent.setup(settings=settings5, accessible_markets_ids=[0, 1, 2])
        assert agent.order_time_length == 1

    @pytest.mark.parametrize("index_price", [350.0, 350.05, 349.95, 351, 349])
    @pytest.mark.parametrize("order_volume", [1, 2])
    def test_submit_orders(self, index_price: float, order_volume: int) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        _prng = random.Random(42)
        agent = ArbitrageAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent", logger=logger
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": order_volume,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        market1 = Market(
            market_id=0,
            prng=random.Random(1),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market1.setup(
            settings={
                "tickSize": 0.01,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        )
        sim._add_market(market=market1, group_name="market")
        market1._is_running = True
        market1._update_time(next_fundamental_price=300.0)
        market2 = Market(
            market_id=1,
            prng=random.Random(1),
            simulator=sim,
            name="market2",
            logger=logger,
        )
        market2.setup(
            settings={
                "tickSize": 0.01,
                "fundamentalPrice": 400.0,
                "outstandingShares": 2000,
            }
        )
        sim._add_market(market=market2, group_name="market")
        market2._is_running = True
        market2._update_time(next_fundamental_price=400.0)
        index_market = IndexMarket(
            market_id=2,
            prng=random.Random(1),
            simulator=sim,
            name="index",
            logger=logger,
        )
        index_market.setup(
            settings={
                "markets": ["market1", "market2"],
                "tickSize": 0.01,
                "fundamentalPrice": index_price,
            }
        )
        index_market._is_running = True
        index_market._update_time(next_fundamental_price=350.0)

        assert index_market.get_index() == 350.0
        assert index_market.get_market_price() == index_price

        orders = agent.submit_orders(markets=[market1, market2, index_market])
        if abs(index_price - 350.0) < 0.1:
            assert len(orders) == 0
        else:
            assert len(orders) == 3
            order1 = cast(Order, [x for x in orders if x.market_id == 0][0])
            order2 = cast(Order, [x for x in orders if x.market_id == 1][0])
            index_order = cast(Order, [x for x in orders if x.market_id == 2][0])
            assert order1.price == 300.0
            assert order2.price == 400.0
            assert index_order.price == index_price
            assert order1.volume == order_volume
            assert order2.volume == order_volume
            assert index_order.volume == 2 * order_volume
            assert order1.ttl == 10
            assert order2.ttl == 10
            assert index_order.ttl == 10
            if index_price > 350.0:
                assert order1.is_buy
                assert order2.is_buy
                assert not index_order.is_buy
            else:
                assert not order1.is_buy
                assert not order2.is_buy
                assert index_order.is_buy
