import random

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Market
from pams import Order
from pams import Session
from pams import Simulator
from pams.events import OrderMistakeShock
from pams.logs import Logger
from tests.pams.events.test_base import TestEventABC


class TestOrderMistakeShock(TestEventABC):
    def test__init__(self) -> None:
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
            "events": ["OrderMistakeShock"],
        }
        session.setup(settings=session_setting)
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
        sim._add_market(market=market)
        _prng = random.Random(42)
        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        assert event.is_enabled
        setting1 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        event.setup(settings=setting1)
        assert event.target_market == market
        assert event.trigger_time == 100
        assert event.price_change_rate == -0.05
        assert event.order_volume == 10000
        assert event.order_time_length == 10000
        assert not event.is_enabled

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting2 = {
            "target": "market1",
            "agent": "aaa",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.warns(Warning):
            event.setup(settings=setting2)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting3 = {
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting3)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting4 = {
            "target": 1,
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting4)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting5 = {
            "target": "market1",
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting5)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting6 = {
            "target": "market1",
            "triggerTime": "error",
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting6)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting7 = {
            "target": "market1",
            "triggerTime": 100,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting7)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting8 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": "error",
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting8)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting9 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting9)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting10 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": "error",
            "orderTimeLength": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting10)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting11 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting11)

        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting12 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": "error",
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting12)

    def test_hook_registration(self) -> None:
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
            "events": ["OrderMistakeShock"],
        }
        session.setup(settings=session_setting)
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
        sim._add_market(market=market)
        _prng = random.Random(42)
        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": True,
        }
        event.setup(settings=setting1)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 1
        event_hook = event_hooks[0]
        assert event_hook.event == event
        assert event_hook.hook_type == "order"
        assert event_hook.is_before
        assert event_hook.time == [100]
        assert event_hook.specific_class is None
        assert event_hook.specific_class is None

        setting2 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": False,
        }
        event.setup(settings=setting2)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 0

    def test_hooked_before_order(self) -> None:
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
            "events": ["OrderMistakeShock"],
        }
        session.setup(settings=session_setting)
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
        sim._add_market(market=market)
        sim.fundamentals.add_market(
            market_id=0, initial=300.0, drift=0.0, volatility=0.0, start_at=0
        )
        _prng = random.Random(42)
        event = OrderMistakeShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": True,
        }
        event.setup(settings=setting1)
        market._update_time(next_fundamental_price=300)
        order = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=1,
            placed_at=None,
            price=None,
            order_id=None,
            ttl=None,
        )
        event.hooked_before_order(simulator=sim, order=order)
        assert order.kind == LIMIT_ORDER
        assert order.price == 300.0 * (1 - 0.05)
        assert not order.is_buy
        assert order.volume == 10000
        assert order.ttl == 10000
        order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=1,
            placed_at=None,
            price=None,
            order_id=None,
            ttl=None,
        )
        event.hooked_before_order(simulator=sim, order=order2)
        assert order2.kind == MARKET_ORDER
        assert order2.price is None
        assert order2.is_buy
        assert order2.volume == 1
        assert order2.ttl is None
