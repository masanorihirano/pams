import random

import pytest

from pams import Market
from pams import Session
from pams import Simulator
from pams.events import EventABC
from pams.events import PriceLimitRule
from pams.logs import Logger
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from tests.pams.events.test_base import TestEventABC


class TestPriceLimitRule(TestEventABC):
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
            "events": ["PriceLimitRule"],
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
        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        assert event.is_enabled
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "enabled": False,
        }
        event.setup(settings=setting1)

        assert "market1" in event.target_markets
        assert market in event.target_markets.values()
        assert event.trigger_change_rate == 0.05
        assert not event.is_enabled

        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting2 = {"triggerChangeRate": 0.05, "enabled": False}
        with pytest.raises(ValueError):
            event.setup(settings=setting2)

        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting3 = {"targetMarkets": ["market1"], "enabled": False}
        with pytest.raises(ValueError):
            event.setup(settings=setting3)

        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting4 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 1,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting4)
        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting5 = {
            "targetMarkets": "market1",
            "triggerChangeRate": 1,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting5)
        setting6 = {"targetMarkets": [0], "triggerChangeRate": 1, "enabled": False}
        with pytest.raises(ValueError):
            event.setup(settings=setting6)
        setting7 = {
            "targetMarkets": ["market2"],
            "triggerChangeRate": 1,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting7)
        setting8 = {
            "referenceMarket": "market1",
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "enabled": False,
        }
        with pytest.warns(Warning):
            event.setup(settings=setting8)
        return event

    def test_get_limited_price(self) -> None:
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
            "events": ["PriceLimitRule"],
        }
        session.setup(settings=session_setting)
        market = Market(
            market_id=0,
            prng=random.Random(42),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market2 = Market(
            market_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="market2",
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
        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "enabled": False,
        }
        event.setup(settings=setting1)
        market._update_time(next_fundamental_price=300.0)
        market2._update_time(next_fundamental_price=300.0)
        dummy_order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10
        )
        with pytest.raises(AssertionError):
            event.get_limited_price(order=dummy_order, market=market2)
        dummy_order2 = Order(
            agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1
        )
        results = event.get_limited_price(order=dummy_order2, market=market)
        assert results is None
        dummy_order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=300.0,
        )
        results = event.get_limited_price(order=dummy_order2, market=market)
        assert results == 300.0

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
            "events": ["PriceLimitRule"],
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
        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "enabled": True,
        }
        event.setup(settings=setting1)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 1
        event_hook = event_hooks[0]
        assert event_hook.event == event
        assert event_hook.hook_type == "order"
        assert event_hook.is_before
        assert event_hook.time is None
        assert event_hook.specific_instance is None
        assert event_hook.specific_class is None

        _prng = random.Random(42)
        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting2 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
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
            "events": ["PriceLimitRule"],
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
        event = PriceLimitRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "enabled": True,
        }
        event.setup(settings=setting1)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10.1
        )
        with pytest.raises(AssertionError):
            event.hooked_before_order(simulator=sim, order=order)
        market._update_time(next_fundamental_price=300.0)
        event.hooked_before_order(simulator=sim, order=order)
        assert order.price == 300.0 * (1.0 - event.trigger_change_rate)
