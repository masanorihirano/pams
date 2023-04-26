import random

import pytest

from pams import Market
from pams import Session
from pams import Simulator
from pams.agents import ArbitrageAgent
from pams.agents import FCNAgent
from pams.events import FundamentalPriceShock
from pams.logs import Logger


class TestSimulator:
    def test__init(self) -> None:
        prng = random.Random(42)
        logger = Logger()
        sim = Simulator(prng=prng, logger=logger)
        assert sim._prng == prng
        assert sim.logger == logger

    def test_add(self) -> None:
        prng = random.Random(42)
        logger = Logger()
        sim = Simulator(prng=prng, logger=logger)
        session = Session(
            session_id=0,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        sim._add_session(session=session)
        with pytest.raises(ValueError):
            sim._add_session(session=session)
        session = Session(
            session_id=0,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        with pytest.raises(ValueError):
            sim._add_session(session=session)
        session = Session(
            session_id=1,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        with pytest.raises(ValueError):
            sim._add_session(session=session)
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        sim._add_market(market=market, group_name="market_group")
        with pytest.raises(ValueError):
            sim._add_market(market=market)
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        with pytest.raises(ValueError):
            sim._add_market(market=market)
        market = Market(
            market_id=1, prng=random.Random(42), simulator=sim, name="market"
        )
        with pytest.raises(ValueError):
            sim._add_market(market=market)
        event = FundamentalPriceShock(
            event_id=0,
            prng=random.Random(),
            session=session,
            simulator=sim,
            name="event1",
        )
        event.setup(
            settings={"target": "market", "triggerTime": 0, "priceChangeRate": -0.1}
        )
        event_hooks = event.hook_registration()
        for event_hook in event_hooks:
            sim._add_event(event_hook)
        with pytest.raises(ValueError):
            sim._add_event(event_hooks[0])
        agent = FCNAgent(
            agent_id=0, prng=random.Random(42), simulator=sim, name="agent"
        )
        sim._add_agent(agent=agent, group_name="agent_group")
        agent2 = ArbitrageAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="arbitrage"
        )
        sim._add_agent(agent=agent2, group_name="agent_group")
        with pytest.raises(ValueError):
            sim._add_agent(agent=agent)
        agent = FCNAgent(
            agent_id=0, prng=random.Random(42), simulator=sim, name="agent"
        )
        with pytest.raises(ValueError):
            sim._add_agent(agent=agent)
        agent = FCNAgent(
            agent_id=2, prng=random.Random(42), simulator=sim, name="agent"
        )
        with pytest.raises(ValueError):
            sim._add_agent(agent=agent)
