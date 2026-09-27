import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pytest

from pams import Cancel
from pams import Market
from pams import Session
from pams import Simulator
from pams.agents import HighFrequencyAgent
from pams.events import EventABC
from pams.events import TradingHaltRule
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import MarketStepBeginLog
from pams.order import LIMIT_ORDER
from pams.order import Order
from pams.runners import SequentialRunner
from tests.pams.events.test_base import TestEventABC


class TestTradingHaltRule(TestEventABC):
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
            "events": ["TradingHaltRule"],
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
        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        assert event.is_enabled
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        event.setup(settings=setting1)

        assert "market1" in event.target_markets
        assert market in event.target_markets.values()
        assert event.trigger_change_rate == 0.05
        assert event.halting_time_length == 100
        assert not event.is_enabled

        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting2 = {
            "triggerChangeRate": 0.05,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting2)

        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting3 = {
            "targetMarkets": ["market1"],
            "haltingTimeLength": 100,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting3)

        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting4 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 1,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting4)
        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting5 = {
            "targetMarkets": "market1",
            "triggerChangeRate": 1,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting5)
        setting6 = {
            "targetMarkets": [0],
            "triggerChangeRate": 1,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting6)
        setting7 = {
            "targetMarkets": ["market2"],
            "triggerChangeRate": 1,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting7)
        setting8 = {
            "referenceMarket": "market1",
            "targetMarkets": ["market1"],
            "haltingTimeLength": 100,
            "triggerChangeRate": 0.05,
            "enabled": False,
        }
        with pytest.warns(Warning):
            event.setup(settings=setting8)
        setting9 = {
            "targetMarkets": ["market1"],
            "haltingTimeLength": "test",
            "triggerChangeRate": 0.05,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting9)
        setting10 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "enabled": False,
        }
        with pytest.raises(ValueError):
            event.setup(settings=setting10)
        return event

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
            "events": ["TradingHaltRule"],
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
        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "haltingTimeLength": 100,
            "enabled": True,
        }
        event.setup(settings=setting1)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 2
        event_hook = event_hooks[0]
        assert event_hook.event == event
        assert event_hook.hook_type == "execution"
        assert not event_hook.is_before
        assert event_hook.time is None
        assert event_hook.specific_instance is None
        assert event_hook.specific_class is None
        event_hook = event_hooks[1]
        assert event_hook.event == event
        assert event_hook.hook_type == "market"
        assert event_hook.is_before
        assert event_hook.time is None
        assert event_hook.specific_instance is market
        assert event_hook.specific_class is None

        _prng = random.Random(42)
        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting2 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "haltingTimeLength": 100,
            "enabled": False,
        }
        event.setup(settings=setting2)
        event_hooks = event.hook_registration()
        assert len(event_hooks) == 0

    def test_hooked(self) -> None:
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
            "events": ["TradingHaltRule"],
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
        event = TradingHaltRule(
            event_id=1, prng=_prng, session=session, simulator=sim, name="event"
        )
        setting1 = {
            "targetMarkets": ["market1"],
            "triggerChangeRate": 0.05,
            "haltingTimeLength": 100,
            "enabled": True,
        }
        event.setup(settings=setting1)
        order1 = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10.1
        )
        order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=10.1,
        )
        execution_log = ExecutionLog(
            market_id=0,
            time=1,
            buy_agent_id=0,
            sell_agent_id=0,
            buy_order_id=0,
            sell_order_id=0,
            price=10.0,
            volume=1,
        )
        market._is_running = True
        market._update_time(next_fundamental_price=300.0)
        market._update_time(next_fundamental_price=300.0)
        market._add_order(order1)
        market._add_order(order2)
        market._execution()

        # while not halting, the before-step hook never changes the running state
        market._is_running = False
        for _ in range(3):
            event.hooked_before_step_for_market(simulator=sim, market=market)
        assert not market.is_running
        market._is_running = True
        event.hooked_before_step_for_market(simulator=sim, market=market)
        assert market.is_running

        sim.current_session = session
        event.hooked_after_execution(simulator=sim, execution_log=execution_log)
        assert session.with_order_execution
        assert not market.is_running
        assert event.is_halting
        assert event.activation_count == 1
        assert event.halting_time_started == market.get_time()
        # further execution logs during the halt are not counted
        event.hooked_after_execution(simulator=sim, execution_log=execution_log)
        assert event.activation_count == 1
        for _ in range(100):
            market._update_time(next_fundamental_price=300.0)
            # e.g., the running state is reset at the beginning of a new session
            market._is_running = True
            event.hooked_before_step_for_market(simulator=sim, market=market)
            assert not market.is_running
        market._update_time(next_fundamental_price=300.0)
        # resuming requires the current session
        sim.current_session = None
        with pytest.raises(AssertionError):
            event.hooked_before_step_for_market(simulator=sim, market=market)
        assert event.is_halting
        assert not market.is_running
        sim.current_session = session
        event.hooked_before_step_for_market(simulator=sim, market=market)
        assert market.is_running
        assert not event.is_halting
        assert session.with_order_execution
        sim.current_session = None
        event.hooked_before_step_for_market(simulator=sim, market=market)
        assert market.is_running

    def test_hooked_multiple_markets(self) -> None:
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
            "events": ["TradingHaltRule"],
        }
        session.setup(settings=session_setting)
        session_no_execution = Session(
            session_id=1,
            prng=random.Random(42),
            session_start_time=500,
            simulator=sim,
            name="session1",
            logger=logger,
        )
        session_no_execution.setup(
            settings={
                "sessionName": 1,
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
            }
        )
        markets = []
        for i in range(3):
            market = Market(
                market_id=i,
                prng=random.Random(42),
                simulator=sim,
                name=f"market{i}",
                logger=logger,
            )
            market.setup(
                settings={
                    "tickSize": 0.01,
                    "marketPrice": 300.0,
                    "outstandingShares": 2000,
                }
            )
            sim._add_market(market=market)
            sim.fundamentals.add_market(
                market_id=i, initial=300.0, drift=0.0, volatility=0.0, start_at=0
            )
            market._is_running = True
            market._update_time(next_fundamental_price=300.0)
            market._update_time(next_fundamental_price=300.0)
            markets.append(market)
        event = TradingHaltRule(
            event_id=1, prng=random.Random(42), session=session, simulator=sim, name="e"
        )
        event.setup(
            settings={
                "targetMarkets": ["market0", "market1"],
                "triggerChangeRate": 0.05,
                "haltingTimeLength": 10,
            }
        )
        for market in markets:
            market._add_order(
                Order(
                    agent_id=0,
                    market_id=market.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=10.0,
                )
            )
            market._add_order(
                Order(
                    agent_id=0,
                    market_id=market.market_id,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=10.0,
                )
            )
            market._execution()
        sim.current_session = session

        def _execution_log(market_id: int) -> ExecutionLog:
            return ExecutionLog(
                market_id=market_id,
                time=2,
                buy_agent_id=0,
                sell_agent_id=0,
                buy_order_id=0,
                sell_order_id=0,
                price=10.0,
                volume=1,
            )

        # executions in a non-target market are ignored
        event.hooked_after_execution(simulator=sim, execution_log=_execution_log(2))
        assert not event.is_halting
        assert all(market.is_running for market in markets)

        # all target markets are halted together
        event.hooked_after_execution(simulator=sim, execution_log=_execution_log(1))
        assert event.is_halting
        assert event.activation_count == 1
        assert not markets[0].is_running
        assert not markets[1].is_running
        assert markets[2].is_running
        assert session.with_order_execution

        # halt expires in a session without execution: markets are not restarted
        sim.current_session = session_no_execution
        for _ in range(11):
            for market in markets:
                market._update_time(next_fundamental_price=300.0)
            for market in markets[:2]:
                event.hooked_before_step_for_market(simulator=sim, market=market)
        assert not event.is_halting
        assert not markets[0].is_running
        assert not markets[1].is_running
        assert not session_no_execution.with_order_execution


class _HaltRecordingLogger(Logger):
    """Record executions and per-step running states of markets."""

    def __init__(self) -> None:
        super().__init__()
        self.execution_logs: List[ExecutionLog] = []
        self.running: Dict[Tuple[str, int], bool] = {}
        self.session_of_time: Dict[int, int] = {}

    def process_execution_log(self, log: ExecutionLog) -> None:
        self.execution_logs.append(log)

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        time = log.market.get_time()
        self.running[(log.market.name, time)] = log.market.is_running
        self.session_of_time[time] = log.session.session_id


class _CrossingHighFrequencyAgent(HighFrequencyAgent):
    """Submit a crossing buy/sell pair to every market."""

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        return [
            Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=1,
                price=300.0,
            )
            for market in markets
            for is_buy in (True, False)
        ]


def _agents_setting(market_name: str) -> Dict[str, Any]:
    return {
        "class": "FCNAgent",
        "numAgents": 30,
        "markets": [market_name],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    }


def _run_halt_simulation(
    market_names: List[str],
    target_markets: List[str],
    halting_time_length: int,
    first_session_execution: bool,
    seed: int = 1,
) -> Tuple[SequentialRunner, _HaltRecordingLogger]:
    config: Dict[str, Any] = {
        "simulation": {
            "markets": market_names,
            "agents": [f"Agents{name}" for name in market_names],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 30,
                    "withOrderPlacement": True,
                    "withOrderExecution": first_session_execution,
                    "withPrint": False,
                },
                {
                    "sessionName": 1,
                    "iterationSteps": 200,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": False,
                    "events": ["FundamentalPriceShock", "TradingHaltRule"],
                },
            ],
        },
        "FundamentalPriceShock": {
            "class": "FundamentalPriceShock",
            "target": target_markets[0],
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "enabled": True,
        },
        "TradingHaltRule": {
            "class": "TradingHaltRule",
            "targetMarkets": target_markets,
            "triggerChangeRate": 0.05,
            "haltingTimeLength": halting_time_length,
            "enabled": True,
        },
    }
    for name in market_names:
        config[name] = {
            "class": "Market",
            "tickSize": 0.00001,
            "marketPrice": 300.0,
            "outstandingShares": 25000,
        }
        config[f"Agents{name}"] = _agents_setting(name)
    logger = _HaltRecordingLogger()
    runner = SequentialRunner(settings=config, prng=random.Random(seed), logger=logger)
    runner.main()
    return runner, logger


def _executions(
    runner: SequentialRunner, logger: _HaltRecordingLogger
) -> List[Tuple[str, int]]:
    return [
        (runner.simulator.id2market[log.market_id].name, log.time)
        for log in logger.execution_logs
    ]


class TestTradingHaltRuleInSimulation:
    def test_no_execution_in_session_without_execution(self) -> None:
        runner, logger = _run_halt_simulation(
            market_names=["MarketA"],
            target_markets=["MarketA"],
            halting_time_length=10,
            first_session_execution=False,
        )
        executions = _executions(runner, logger)
        session0 = runner.simulator.sessions[0]
        assert not session0.with_order_execution
        assert runner.simulator.sessions[1].with_order_execution
        executions_in_session0 = [
            t for _, t in executions if logger.session_of_time[t] == 0
        ]
        assert executions_in_session0 == []
        assert not any(
            running
            for (_, t), running in logger.running.items()
            if logger.session_of_time[t] == 0
        )
        assert len(executions) > 0

    def _halted_times(
        self, logger: _HaltRecordingLogger, market_name: str
    ) -> List[int]:
        return sorted(
            t
            for (name, t), running in logger.running.items()
            if name == market_name and not running
        )

    def test_non_target_market_keeps_executing(self) -> None:
        runner, logger = _run_halt_simulation(
            market_names=["MarketA", "MarketB"],
            target_markets=["MarketA"],
            halting_time_length=10,
            first_session_execution=True,
        )
        executions = _executions(runner, logger)
        event = runner.simulator.name2event["TradingHaltRule"]
        assert isinstance(event, TradingHaltRule)
        assert event.activation_count >= 1
        halted_a = self._halted_times(logger, "MarketA")
        assert len(halted_a) > 0
        assert self._halted_times(logger, "MarketB") == []
        halted_set = set(halted_a)
        assert not any(name == "MarketA" and t in halted_set for name, t in executions)
        assert any(name == "MarketB" and t in halted_set for name, t in executions)
        for session in runner.simulator.sessions:
            assert session.with_order_execution

    def test_multiple_target_markets_halt_and_resume_together(self) -> None:
        runner, logger = _run_halt_simulation(
            market_names=["MarketA", "MarketB", "MarketC"],
            target_markets=["MarketA", "MarketB"],
            halting_time_length=10,
            first_session_execution=True,
        )
        executions = _executions(runner, logger)
        event = runner.simulator.name2event["TradingHaltRule"]
        assert isinstance(event, TradingHaltRule)
        assert event.activation_count >= 1
        halted_a = self._halted_times(logger, "MarketA")
        assert len(halted_a) > 0
        assert halted_a == self._halted_times(logger, "MarketB")
        assert self._halted_times(logger, "MarketC") == []
        # each halt lasts haltingTimeLength steps after the triggering step
        assert len(halted_a) == event.activation_count * event.halting_time_length
        halted_set = set(halted_a)
        assert not any(
            name in ("MarketA", "MarketB") and t in halted_set for name, t in executions
        )
        assert any(name == "MarketC" and t in halted_set for name, t in executions)

    def test_resume_within_executing_session(self) -> None:
        runner, logger = _run_halt_simulation(
            market_names=["MarketA"],
            target_markets=["MarketA"],
            halting_time_length=10,
            first_session_execution=True,
        )
        executions = _executions(runner, logger)
        halted_a = self._halted_times(logger, "MarketA")
        assert len(halted_a) > 0
        last_halted = halted_a[-1]
        assert logger.running[("MarketA", last_halted + 1)]
        assert any(name == "MarketA" and t > last_halted for name, t in executions)

    def test_high_frequency_execution_respects_halt(self) -> None:
        config: Dict[str, Any] = {
            "simulation": {
                "markets": ["MarketA", "MarketB"],
                "agents": ["HFAgents"],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 10,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": False,
                        "maxHighFrequencyOrders": 1,
                        "highFrequencySubmitRate": 1.0,
                    }
                ],
            },
            "MarketA": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "MarketB": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "HFAgents": {
                "class": "_CrossingHighFrequencyAgent",
                "numAgents": 1,
                "markets": ["MarketA", "MarketB"],
                "assetVolume": 50,
                "cashAmount": 100000,
            },
        }
        logger = _HaltRecordingLogger()
        runner = SequentialRunner(
            settings=config, prng=random.Random(42), logger=logger
        )
        runner.class_register(_CrossingHighFrequencyAgent)
        runner._setup()
        session = runner.simulator.sessions[0]
        market_a = runner.simulator.name2market["MarketA"]
        market_b = runner.simulator.name2market["MarketB"]
        for market in runner.simulator.markets:
            market._update_time(next_fundamental_price=300.0)
        market_a._is_running = False  # e.g., halted by TradingHaltRule
        market_b._is_running = True
        # a normal order (without counterpart) followed by high frequency orders
        local_orders: List[List[Union[Order, Cancel]]] = [
            [
                Order(
                    agent_id=0,
                    market_id=market_b.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=100.0,
                )
            ]
        ]
        runner._handle_orders(session=session, local_orders=local_orders)
        runner.logger._process()  # type: ignore
        executions = _executions(runner, logger)
        assert ("MarketA", 0) not in executions
        assert ("MarketB", 0) in executions
        assert session.with_order_execution
