import random
from typing import Optional
from typing import Type

import pytest

from pams import Market
from pams import Session
from pams import Simulator
from pams.events import EventHook
from pams.events import FundamentalPriceShock
from pams.logs import Logger
from tests.pams.agents.test_base import DummyAgent


class TestEventHook:
    @pytest.mark.parametrize(
        "hook_type", ["order", "cancel", "execution", "session", "market", "dummy"]
    )
    @pytest.mark.parametrize("is_before", [True, False])
    @pytest.mark.parametrize("specific_class", [None, Market, DummyAgent])
    @pytest.mark.parametrize("specific_instance_name", [None, "market", "agent"])
    def test(
        self,
        hook_type: str,
        is_before: bool,
        specific_class: Optional[Type],
        specific_instance_name: Optional[str],
    ) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        session = Session(
            session_id=0,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session0",
            logger=logger,
        )
        session_setting = {
            "sessionName": 0,
            "iterationSteps": 500,
            "withOrderPlacement": True,
            "withOrderExecution": True,
            "withPrint": True,
            "maxNormalOrders": 1,
            "events": ["FundamentalPriceShock"],
        }
        session.setup(settings=session_setting)
        sim._add_session(session=session)
        market = Market(
            market_id=0,
            prng=random.Random(42),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        settings_market = {
            "tickSize": 0.01,
            "marketPrice": 300.0,
            "outstandingShares": 2000,
        }
        market.setup(settings=settings_market)
        agent = DummyAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings_agent = {"assetVolume": 50, "cashAmount": 10000}
        agent.setup(settings=settings_agent, accessible_markets_ids=[0])
        sim._add_agent(agent=agent)

        event = FundamentalPriceShock(
            event_id=0,
            prng=random.Random(42),
            session=session,
            simulator=sim,
            name="event0",
        )
        specific_instance: Optional[object]
        if specific_instance_name is None:
            specific_instance = None
        elif specific_instance_name == "market":
            specific_instance = market
        elif specific_instance_name == "agent":
            specific_instance = agent
        else:
            raise NotImplementedError
        if (
            (hook_type == "execution" and is_before)
            or hook_type == "dummy"
            or (hook_type != "market" and specific_class is not None)
            or (hook_type != "market" and specific_instance is not None)
            or specific_class == DummyAgent
            or specific_instance_name == "agent"
        ):
            with pytest.raises(ValueError):
                EventHook(
                    event=event,
                    hook_type=hook_type,
                    is_before=is_before,
                    time=[1, 3],
                    specific_class=specific_class,
                    specific_instance=specific_instance,
                )
            return
        else:
            event_hook = EventHook(
                event=event,
                hook_type=hook_type,
                is_before=is_before,
                time=[1, 3],
                specific_class=specific_class,
                specific_instance=specific_instance,
            )
        sim._add_event(event_hook=event_hook)
        assert event_hook.event == event
        assert event_hook.hook_type == hook_type
        assert event_hook.is_before == is_before
        assert event_hook.time == [1, 3]
        assert event_hook.specific_class == specific_class
        assert event_hook.specific_instance == specific_instance
        assert (
            str(event_hook)
            == f"<pams.events.base.EventHook | hook_type={hook_type}, is_before={is_before}, time=[1, 3], "
            f"specific_class={specific_class}, specific_instance={specific_instance}, event={event}>"
        )
