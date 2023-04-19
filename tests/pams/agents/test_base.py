import random
from typing import List
from typing import Union

import pytest

from pams import Cancel
from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import Agent
from pams.logs import Logger


class DummyAgent(Agent):
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        return []


class TestAgent:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        agent = DummyAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        assert agent.agent_id == 1
        assert agent.prng.random() == random.Random(42).random()
        assert agent.name == "test_agent"
        assert len(agent.asset_volumes) == 0
        assert agent.cash_amount == 0
        assert agent.simulator == sim
        assert agent.logger == logger

        assert (
            str(agent)
            == f"<tests.pams.agents.test_base.DummyAgent | id=1, name=test_agent, logger={logger}>"
        )

    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        agent = DummyAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings1 = {"assetVolume": 50, "cashAmount": 10000}
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

        agent = DummyAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        _prng = random.Random(42)
        settings2 = {"assetVolume": [40, 60], "cashAmount": [9000, 10000]}
        agent.setup(settings=settings2, accessible_markets_ids=[1, 2, 3])
        expected_cash_amount = _prng.random() * 1000 + 9000
        assert agent.cash_amount == expected_cash_amount
        expect_asset_volumes = {
            1: int(_prng.random() * 20 + 40),
            2: int(_prng.random() * 20 + 40),
            3: int(_prng.random() * 20 + 40),
        }
        assert agent.asset_volumes == expect_asset_volumes
        assert not agent.is_market_accessible(0)
        assert agent.is_market_accessible(1)
        assert agent.is_market_accessible(2)
        assert agent.is_market_accessible(3)
        assert agent.get_asset_volume(1) == expect_asset_volumes[1]
        assert agent.get_asset_volume(2) == expect_asset_volumes[2]
        assert agent.get_asset_volume(3) == expect_asset_volumes[3]
        with pytest.raises(ValueError):
            agent.get_asset_volume(0)
        assert agent.get_cash_amount() == expected_cash_amount
        assert agent.get_prng().random() == _prng.random()

        agent.set_asset_volume(market_id=1, volume=100)
        assert agent.get_asset_volume(market_id=1) == 100
        with pytest.raises(ValueError):
            agent.set_asset_volume(market_id=0, volume=100)
        with pytest.raises(ValueError):
            agent.set_asset_volume(market_id=1, volume=10.1)  # type: ignore  # for error testing

        agent.set_cash_amount(cash_amount=1.1)
        assert agent.get_cash_amount() == 1.1

        agent.set_market_accessible(market_id=0)
        assert agent.is_market_accessible(0)
        with pytest.raises(ValueError):
            agent.set_market_accessible(market_id=1)

        volume = agent.get_asset_volume(market_id=1)
        agent.update_asset_volume(market_id=1, delta=-100)
        assert agent.get_asset_volume(market_id=1) == volume - 100
        with pytest.raises(ValueError):
            agent.update_asset_volume(market_id=4, delta=1)
        with pytest.raises(ValueError):
            agent.update_asset_volume(market_id=2, delta=1.1)  # type: ignore  # for error testing

        cash = agent.get_cash_amount()
        agent.update_cash_amount(delta=101.23)
        assert agent.get_cash_amount() == cash + 101.23

        agent = DummyAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings3 = {"cashAmount": [9000, 10000]}
        with pytest.raises(ValueError):
            agent.setup(settings=settings3, accessible_markets_ids=[1, 2, 3])
        agent = DummyAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings4 = {"assetVolume": 50}
        with pytest.raises(ValueError):
            agent.setup(settings=settings4, accessible_markets_ids=[1, 2, 3])
