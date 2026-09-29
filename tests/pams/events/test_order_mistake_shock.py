import math
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Market
from pams import Order
from pams import Session
from pams import Simulator
from pams.events import EventABC
from pams.events import OrderMistakeShock
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import MarketStepBeginLog
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
from pams.runners import SequentialRunner
from tests.pams.events.test_base import TestEventABC


class _ShockRecordingLogger(Logger):
    """Record orders, executions, and the order books of each step."""

    def __init__(self) -> None:
        super().__init__()
        self.order_logs: List[OrderLog] = []
        self.execution_logs: List[ExecutionLog] = []
        self.step_begin_prices: Dict[int, float] = {}
        self.step_end_order_ids: Dict[int, Set[int]] = {}

    def process_order_log(self, log: OrderLog) -> None:
        self.order_logs.append(log)

    def process_execution_log(self, log: ExecutionLog) -> None:
        self.execution_logs.append(log)

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        self.step_begin_prices[log.market.get_time()] = log.market.get_market_price()

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        market = log.market
        order_ids = {order.order_id for order in market.buy_order_book.priority_queue}
        order_ids |= {order.order_id for order in market.sell_order_book.priority_queue}
        self.step_end_order_ids[market.get_time()] = {
            order_id for order_id in order_ids if order_id is not None
        }


class TestOrderMistakeShock(TestEventABC):
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
        return event

    @pytest.mark.parametrize(
        "price_change_rate, order_volume, message",
        [
            (-1.5, 10000, "priceChangeRate have to be greater than -1.0"),
            (-1.0, 10000, "priceChangeRate have to be greater than -1.0"),
            (-0.05, 0, "orderVolume have to be positive"),
            (-0.05, -5, "orderVolume have to be positive"),
            (-0.99, 1, None),
        ],
    )
    def test_setup_range(
        self, price_change_rate: float, order_volume: int, message: Optional[str]
    ) -> None:
        # plhamJ silently skips a mistaken order with a negative price or a non-positive
        # volume, but such settings are rejected at the setup
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
        market = Market(
            market_id=0,
            prng=random.Random(42),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market.setup(
            settings={"tickSize": 0.01, "marketPrice": 300.0, "outstandingShares": 2000}
        )
        sim._add_market(market=market)
        event = OrderMistakeShock(
            event_id=1,
            prng=random.Random(42),
            session=session,
            simulator=sim,
            name="event",
        )
        settings = {
            "target": "market1",
            "triggerTime": 100,
            "priceChangeRate": price_change_rate,
            "orderVolume": order_volume,
            "orderTimeLength": 10,
        }
        if message is None:
            event.setup(settings=settings)
            assert event.price_change_rate == price_change_rate
            assert event.order_volume == order_volume
        else:
            with pytest.raises(ValueError, match=message):
                event.setup(settings=settings)

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

    @pytest.mark.parametrize(
        "price_change_rate, is_buy", [(0.05, True), (0.0, False), (-0.05, False)]
    )
    def test_hooked_before_order_side(
        self, price_change_rate: float, is_buy: bool
    ) -> None:
        # as OrderMistakeShock of plhamJ, a non-positive rate places a sell order
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
        session.setup(
            settings={
                "sessionName": 0,
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "events": ["OrderMistakeShock"],
            }
        )
        market = Market(
            market_id=0,
            prng=random.Random(42),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market.setup(
            settings={"tickSize": 0.01, "marketPrice": 300.0, "outstandingShares": 2000}
        )
        sim._add_market(market=market)
        sim.fundamentals.add_market(
            market_id=0, initial=300.0, drift=0.0, volatility=0.0, start_at=0
        )
        market._update_time(next_fundamental_price=300.0)
        event = OrderMistakeShock(
            event_id=1,
            prng=random.Random(42),
            session=session,
            simulator=sim,
            name="event",
        )
        event.setup(
            settings={
                "target": "market1",
                "triggerTime": 100,
                "priceChangeRate": price_change_rate,
                "orderVolume": 10000,
                "orderTimeLength": 10,
            }
        )
        order = Order(
            agent_id=0,
            market_id=0,
            is_buy=not is_buy,
            kind=MARKET_ORDER,
            volume=1,
            placed_at=None,
            price=None,
            order_id=None,
            ttl=None,
        )
        event.hooked_before_order(simulator=sim, order=order)
        assert order.is_buy == is_buy
        assert order.kind == LIMIT_ORDER
        assert order.price == pytest.approx(300.0 * (1 + price_change_rate))
        assert order.volume == 10000
        assert order.ttl == 10

    def test_hooked_before_order_other_market(self) -> None:
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
        for market_id, name, price in [(0, "market1", 300.0), (1, "market2", 500.0)]:
            market = Market(
                market_id=market_id,
                prng=random.Random(42),
                simulator=sim,
                name=name,
                logger=logger,
            )
            market.setup(
                settings={
                    "tickSize": 0.01,
                    "marketPrice": price,
                    "outstandingShares": 2000,
                }
            )
            sim._add_market(market=market)
            sim.fundamentals.add_market(
                market_id=market_id,
                initial=price,
                drift=0.0,
                volatility=0.0,
                start_at=0,
            )
            market._update_time(next_fundamental_price=price)
        event = OrderMistakeShock(
            event_id=1,
            prng=random.Random(42),
            session=session,
            simulator=sim,
            name="event",
        )
        setting = {
            "target": "market2",
            "triggerTime": 100,
            "priceChangeRate": -0.05,
            "orderVolume": 10000,
            "orderTimeLength": 10000,
            "enabled": True,
        }
        event.setup(settings=setting)
        # an order to a non-target market must not be overridden
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
        assert order.kind == MARKET_ORDER
        assert order.price is None
        assert order.is_buy
        assert order.volume == 1
        assert order.ttl is None
        assert not event.triggerd
        # the first order to the target market is overridden
        order2 = Order(
            agent_id=0,
            market_id=1,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=1,
            placed_at=None,
            price=None,
            order_id=None,
            ttl=None,
        )
        event.hooked_before_order(simulator=sim, order=order2)
        assert order2.kind == LIMIT_ORDER
        assert order2.price == 500.0 * (1 - 0.05)
        assert not order2.is_buy
        assert order2.volume == 10000
        assert order2.ttl == 10000
        assert event.triggerd
        # the shock is applied only once
        order3 = Order(
            agent_id=0,
            market_id=1,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=1,
            placed_at=None,
            price=None,
            order_id=None,
            ttl=None,
        )
        event.hooked_before_order(simulator=sim, order=order3)
        assert order3.kind == MARKET_ORDER
        assert order3.volume == 1

    def test_target_market_in_runner(self) -> None:
        class OrderLogCollector(Logger):
            def __init__(self) -> None:
                super().__init__()
                self.order_logs: List[OrderLog] = []

            def process_order_log(self, log: OrderLog) -> None:
                self.order_logs.append(log)

        config = {
            "simulation": {
                "markets": ["Market"],
                "agents": ["FCNAgents"],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 10,
                        "withOrderPlacement": True,
                        "withOrderExecution": False,
                        "withPrint": False,
                    },
                    {
                        "sessionName": 1,
                        "iterationSteps": 20,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": False,
                        "events": ["OrderMistakeShock"],
                    },
                ],
            },
            "OrderMistakeShock": {
                "class": "OrderMistakeShock",
                "target": "Market-1",
                "triggerTime": 5,
                "priceChangeRate": -0.05,
                "orderVolume": 10000,
                "orderTimeLength": 10000,
                "enabled": True,
            },
            "Market": {
                "class": "Market",
                "numMarkets": 2,
                "tickSize": 0.00001,
                "marketPrice": 300.0,
                "outstandingShares": 25000,
            },
            "FCNAgents": {
                "class": "FCNAgent",
                "numAgents": 10,
                "markets": ["Market"],
                "assetVolume": 50,
                "cashAmount": 10000,
                "fundamentalWeight": {"expon": [1.0]},
                "chartWeight": {"expon": [0.0]},
                "noiseWeight": {"expon": [1.0]},
                "noiseScale": 0.001,
                "timeWindowSize": [100, 200],
                "orderMargin": [0.0, 0.1],
            },
        }
        logger = OrderLogCollector()
        runner = SequentialRunner(
            settings=config, prng=random.Random(42), logger=logger
        )
        runner.main()
        target_market = runner.simulator.name2market["Market-1"]
        shock_logs = [log for log in logger.order_logs if log.volume == 10000]
        assert len(shock_logs) == 1
        assert shock_logs[0].market_id == target_market.market_id
        assert shock_logs[0].time == 10 + 5
        assert shock_logs[0].kind == LIMIT_ORDER

    @pytest.mark.parametrize(
        "price_change_rate, is_buy, round_to_tick",
        [(-0.0505, False, math.ceil), (0.0505, True, math.floor)],
    )
    def test_mistaken_order_in_runner(
        self,
        price_change_rate: float,
        is_buy: bool,
        round_to_tick: Callable[[float], int],
    ) -> None:
        # as OrderMistakeShock of plhamJ, the mistaken order is placed at the trigger time
        # counted from the session start, its price is rounded to the tick size (up for
        # sell and down for buy), it is executed against the order book immediately, and
        # it expires after orderTimeLength steps
        tick_size = 1.0
        order_time_length = 3
        config = {
            "simulation": {
                "markets": ["Market"],
                "agents": ["FCNAgents"],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 30,
                        "withOrderPlacement": True,
                        "withOrderExecution": False,
                        "withPrint": False,
                    },
                    {
                        "sessionName": 1,
                        "iterationSteps": 10,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": False,
                        "events": ["OrderMistakeShock"],
                    },
                ],
            },
            "OrderMistakeShock": {
                "class": "OrderMistakeShock",
                "target": "Market",
                "triggerTime": 0,
                "priceChangeRate": price_change_rate,
                "orderVolume": 10000,
                "orderTimeLength": order_time_length,
            },
            "Market": {
                "class": "Market",
                "tickSize": tick_size,
                "marketPrice": 300.0,
                "outstandingShares": 25000,
            },
            "FCNAgents": {
                "class": "FCNAgent",
                "numAgents": 10,
                "markets": ["Market"],
                "assetVolume": 50,
                "cashAmount": 10000,
                "fundamentalWeight": {"expon": [1.0]},
                "chartWeight": {"expon": [0.0]},
                "noiseWeight": {"expon": [1.0]},
                "noiseScale": 0.001,
                "timeWindowSize": [100, 200],
                "orderMargin": [0.0, 0.1],
            },
        }
        logger = _ShockRecordingLogger()
        runner = SequentialRunner(
            settings=config, prng=random.Random(42), logger=logger
        )
        runner.main()
        trigger_time = 30
        shock_logs = [log for log in logger.order_logs if log.volume == 10000]
        assert len(shock_logs) == 1
        shock_log = shock_logs[0]
        assert shock_log.time == trigger_time
        assert shock_log.kind == LIMIT_ORDER
        assert shock_log.is_buy == is_buy
        assert shock_log.ttl == order_time_length
        base_price = logger.step_begin_prices[trigger_time]
        expected_price = (
            round_to_tick(base_price * (1 + price_change_rate) / tick_size) * tick_size
        )
        assert shock_log.price == expected_price
        shock_executions = [
            log
            for log in logger.execution_logs
            if shock_log.order_id in (log.buy_order_id, log.sell_order_id)
        ]
        assert len(shock_executions) > 0
        assert shock_executions[0].time == trigger_time
        for log in shock_executions:
            owner = log.buy_agent_id if is_buy else log.sell_agent_id
            assert owner == shock_log.agent_id
        assert sum(log.volume for log in shock_executions) < 10000
        for t in range(trigger_time, trigger_time + order_time_length + 1):
            assert shock_log.order_id in logger.step_end_order_ids[t]
        expired_at = trigger_time + order_time_length + 1
        assert shock_log.order_id not in logger.step_end_order_ids[expired_at]
