import random
from typing import List
from typing import Optional
from typing import Type

import pytest

from pams import LIMIT_ORDER
from pams import Cancel
from pams import Market
from pams import Order
from pams import Session
from pams import Simulator
from pams.events import EventABC
from pams.events import EventHook
from pams.events import FundamentalPriceShock
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import OrderLog
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


class DummyEvent(EventABC):
    def hook_registration(self) -> List[EventHook]:
        event_hooks = []
        for hook_type in ["order", "cancel", "execution", "session", "market"]:
            for is_before in [True, False]:
                if hook_type == "execution" and is_before:
                    continue
                for time in [None, [0, 1, 2]]:
                    event_hook = EventHook(
                        event=self, hook_type=hook_type, is_before=is_before, time=time
                    )
                    event_hooks.append(event_hook)
        return event_hooks


class TestEventABC:
    def test__init__(self) -> EventABC:
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
        _prng = random.Random(42)
        event = DummyEvent(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        assert event.event_id == 1
        assert event.prng == _prng
        assert event.simulator == sim
        assert event.name == "event"
        assert event.session == session
        assert (
            str(event)
            == f"<tests.pams.events.test_base.DummyEvent | id=1, name=event, session={session}>"
        )
        event.setup(settings={})
        event_hooks = event.hook_registration()
        for event_hook in event_hooks:
            sim._add_event(event_hook=event_hook)
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        sim._add_market(market=market, group_name="market_group")
        order = Order(
            agent_id=1,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            placed_at=1,
            price=100.0,
            order_id=1,
        )
        sim._trigger_event_before_order(order=order)
        order_log = OrderLog(
            order_id=order.order_id,  # type: ignore
            market_id=order.market_id,
            time=order.placed_at,  # type: ignore
            agent_id=order.agent_id,
            is_buy=order.is_buy,
            kind=order.kind,
            volume=order.volume,
            price=order.price,
            ttl=order.ttl,
        )
        sim._trigger_event_after_order(order_log=order_log)
        cancel = Cancel(order=order, placed_at=1)
        sim._trigger_event_before_cancel(cancel=cancel)
        cancel_log = CancelLog(
            order_id=cancel.order.order_id,  # type: ignore
            market_id=cancel.order.market_id,
            cancel_time=1,
            order_time=cancel.order.placed_at,  # type: ignore
            agent_id=cancel.agent_id,
            is_buy=cancel.order.is_buy,
            kind=cancel.order.kind,
            volume=cancel.order.volume,
            price=cancel.order.price,
            ttl=cancel.order.ttl,
        )
        sim._trigger_event_after_cancel(cancel_log=cancel_log)
        execution_log = ExecutionLog(
            market_id=0,
            time=1,
            buy_agent_id=0,
            sell_agent_id=1,
            buy_order_id=110,
            sell_order_id=111,
            price=90.9,
            volume=2,
        )
        sim._trigger_event_after_execution(execution_log=execution_log)
        sim._trigger_event_before_session(session=session)
        sim._trigger_event_after_session(session=session)
        market._update_time(next_fundamental_price=300.0)
        sim._trigger_event_before_step_for_market(market=market)
        sim._trigger_event_after_step_for_market(market=market)
        return event
