import random

import pytest

from pams import Market
from pams import Session
from pams import Simulator
from pams.events import FundamentalPriceShock
from pams.logs import Logger
from tests.pams.events.test_base import TestEventABC


class TestFundamentalPriceShock(TestEventABC):
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
            "events": ["FundamentalPriceShock"],
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
        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        assert event.is_enabled
        assert event.shock_time_length == 1
        setting1 = {
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": False,
        }
        event.setup(settings=setting1)
        assert event.target_market_name == "market1"
        assert event.trigger_time == 0
        assert event.price_change_rate == -0.1
        assert not event.is_enabled
        assert event.shock_time_length == 2
        assert event.target_market == market

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting2 = {
            "triggerDays": 100,
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting2)

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting3 = {
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting3)

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting4 = {
            "target": "market1",
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting4)

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting5 = {
            "target": "market1",
            "triggerTime": 0,
            "shockTimeLength": 2,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting5)

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting6 = {
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
        }
        event.setup(settings=setting6)
        assert event.is_enabled

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting7 = {"target": "market1", "triggerTime": 0, "priceChangeRate": -0.1}
        event.setup(settings=setting7)
        assert event.shock_time_length == 1

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting8 = {
            "triggerDays": 100,
            "target": "market1",
            "triggerTime": 0.1,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting8)

        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting9 = {
            "triggerDays": 100,
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2.1,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting9)

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
            "events": ["FundamentalPriceShock"],
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
        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": True,
        }
        event.setup(settings=setting1)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 1
        event_hook = event_hooks[0]
        assert event_hook.event == event
        assert event_hook.hook_type == "market"
        assert event_hook.is_before == True
        assert event_hook.time == [0, 1]
        assert event_hook.specific_instance == market
        assert event_hook.specific_class == None

        setting2 = {
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": False,
        }
        event.setup(settings=setting2)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 0

    def test_hooked_before_step_for_market(self) -> None:
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
        event = FundamentalPriceShock(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "target": "market1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 2,
            "enabled": True,
        }
        event.setup(settings=setting1)
        with pytest.raises(AssertionError):
            event.hooked_before_step_for_market(simulator=sim, market=market)
        market._update_time(next_fundamental_price=300.0)
        event.hooked_before_step_for_market(simulator=sim, market=market)
        assert market.get_fundamental_price() == 300.0 * 0.9

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
        with pytest.raises(AssertionError):
            event.hooked_before_step_for_market(simulator=sim, market=market2)
