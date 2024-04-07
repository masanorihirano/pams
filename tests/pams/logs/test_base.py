import random
from typing import List

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Market
from pams import Session
from pams import Simulator
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import ExpirationLog
from pams.logs import Log
from pams.logs import Logger
from pams.logs import MarketStepBeginLog
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
from pams.logs import SessionBeginLog
from pams.logs import SessionEndLog
from pams.logs import SimulationBeginLog
from pams.logs import SimulationEndLog


class TestLog:
    def test_read_and_write(self) -> None:
        logger = Logger()
        setattr(logger, "count", 0)

        def _write(log: Log) -> None:
            logger.count += 1  # type: ignore

        logger.write = _write  # type: ignore

        log = Log()
        log.read_and_write(logger=logger)

        assert logger.count == 1  # type: ignore

    def test_read_and_write_with_direct_process(self) -> None:
        logger = Logger()
        setattr(logger, "count", 0)

        def _process(logs: List[Log]) -> None:
            logger.count += 1  # type: ignore

        logger.process = _process  # type: ignore

        log = Log()
        log.read_and_write_with_direct_process(logger=logger)

        assert logger.count == 1  # type: ignore


class TestOrderLog:
    def test__init__(self) -> None:
        log = OrderLog(
            order_id=10,
            market_id=1,
            time=2,
            agent_id=3,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=10,
            price=301.0,
            ttl=5,
        )
        assert log.order_id == 10
        assert log.market_id == 1
        assert log.time == 2
        assert log.agent_id == 3
        assert not log.is_buy
        assert log.kind == LIMIT_ORDER
        assert log.volume == 10
        assert log.price == 301.0
        assert log.ttl == 5
        log = OrderLog(
            order_id=9,
            market_id=0,
            time=1,
            agent_id=2,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=9,
        )
        assert log.order_id == 9
        assert log.market_id == 0
        assert log.time == 1
        assert log.agent_id == 2
        assert log.is_buy
        assert log.kind == MARKET_ORDER
        assert log.volume == 9
        assert log.price is None
        assert log.ttl is None


class TestCancelLog:
    def test__init__(self) -> None:
        log = CancelLog(
            order_id=2,
            market_id=3,
            cancel_time=10,
            order_time=8,
            agent_id=4,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=10,
            price=100.0,
            ttl=1,
        )
        assert log.order_id == 2
        assert log.market_id == 3
        assert log.cancel_time == 10
        assert log.order_time == 8
        assert log.agent_id == 4
        assert log.is_buy
        assert log.kind == LIMIT_ORDER
        assert log.volume == 10
        assert log.price == 100.0
        assert log.ttl == 1

        log = CancelLog(
            order_id=2,
            market_id=3,
            cancel_time=10,
            order_time=8,
            agent_id=4,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=10,
        )
        assert log.order_id == 2
        assert log.market_id == 3
        assert log.cancel_time == 10
        assert log.order_time == 8
        assert log.agent_id == 4
        assert log.is_buy
        assert log.kind == MARKET_ORDER
        assert log.volume == 10
        assert log.price is None
        assert log.ttl is None


class TestExpirationLog:
    def test__init__(self) -> None:
        log = ExpirationLog(
            order_id=2,
            market_id=3,
            time=10,
            order_time=8,
            agent_id=4,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=10,
            price=100.0,
            ttl=2,
        )
        assert log.order_id == 2
        assert log.market_id == 3
        assert log.time == 10
        assert log.order_time == 8
        assert log.agent_id == 4
        assert log.is_buy == True
        assert log.kind is LIMIT_ORDER
        assert log.volume == 10
        assert log.price == 100.0
        assert log.ttl == 2

        log = ExpirationLog(
            order_id=2,
            market_id=3,
            time=10,
            order_time=8,
            agent_id=4,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=10,
        )
        assert log.order_id == 2
        assert log.market_id == 3
        assert log.time == 10
        assert log.order_time == 8
        assert log.agent_id == 4
        assert log.is_buy == True
        assert log.kind is MARKET_ORDER
        assert log.volume == 10
        assert log.price is None
        assert log.ttl is None


class TestExecutionLog:
    def test___init__(self) -> None:
        log = ExecutionLog(
            market_id=2,
            time=20,
            buy_agent_id=1,
            sell_agent_id=2,
            buy_order_id=3,
            sell_order_id=4,
            price=101.1,
            volume=2,
        )
        assert log.market_id == 2
        assert log.time == 20
        assert log.buy_agent_id == 1
        assert log.sell_agent_id == 2
        assert log.buy_order_id == 3
        assert log.sell_order_id == 4
        assert log.price == 101.1
        assert log.volume == 2


class TestSimulationBeginLog:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(42))
        log = SimulationBeginLog(simulator=sim)
        assert log.simulator == sim


class TestSimulationEndLog:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(42))
        log = SimulationEndLog(simulator=sim)
        assert log.simulator == sim


class TestSessionBeginLog:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        log = SessionBeginLog(session=session, simulator=sim)
        assert log.session == session
        assert log.simulator == sim


class TestSessionEndLog:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        log = SessionEndLog(session=session, simulator=sim)
        assert log.session == session
        assert log.simulator == sim


class TestMarketStepBeginLog:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        market = Market(
            market_id=2, prng=random.Random(22), simulator=sim, name="test_market"
        )
        log = MarketStepBeginLog(session=session, market=market, simulator=sim)
        assert log.session == session
        assert log.market == market
        assert log.simulator == sim


class TestMarketStepEndLog:
    def test__init__(self) -> None:
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        market = Market(
            market_id=2, prng=random.Random(22), simulator=sim, name="test_market"
        )
        log = MarketStepEndLog(session=session, market=market, simulator=sim)
        assert log.session == session
        assert log.market == market
        assert log.simulator == sim


class TestLogger:
    def test__init__(self) -> None:
        logger = Logger()
        assert logger.pending_logs == []

    def test_set_simulator(self) -> None:
        sim = Simulator(prng=random.Random(42))
        logger = Logger()
        logger._set_simulator(simulator=sim)
        assert logger.simulator == sim

    def test_write(self) -> None:
        logger = Logger()
        log = Log()
        logger.write(log=log)
        assert logger.pending_logs == [log]

    def test_bulk_write(self) -> None:
        logger = Logger()
        log1 = Log()
        log2 = Log()
        log3 = Log()
        logger.bulk_write(logs=[log1, log2, log3])
        assert logger.pending_logs == [log1, log2, log3]

    def test_write_and_direct_process(self) -> None:
        log1 = Log()

        logger = Logger()
        setattr(logger, "count", 0)

        def _process(logs: List[Log]) -> None:
            logger.count += 1  # type: ignore
            assert logs == [log1]

        logger.process = _process  # type: ignore

        logger.write_and_direct_process(log=log1)

        assert logger.count == 1  # type: ignore

    def test_bulk_write_and_direct_process(self) -> None:
        log1 = Log()
        log2 = Log()

        logger = Logger()
        setattr(logger, "count", 0)

        def _process(logs: List[Log]) -> None:
            logger.count += 1  # type: ignore
            assert logs == [log1, log2]

        logger.process = _process  # type: ignore

        logger.bulk_write_and_direct_process(logs=[log1, log2])

        assert logger.count == 1  # type: ignore

    def test__process(self) -> None:
        log1 = Log()
        log2 = Log()

        logger = Logger()
        setattr(logger, "count", 0)

        def _process(logs: List[Log]) -> None:
            logger.count += 1  # type: ignore
            assert logs == [log1, log2]

        logger.process = _process  # type: ignore

        logger.write(log=log1)
        logger.write(log=log2)

        logger._process()

        assert logger.count == 1  # type: ignore
        assert logger.pending_logs == []

    def test_process(self) -> None:
        class DummyLogger(Logger):
            def __init__(self) -> None:
                super().__init__()
                self.n_order_log = 0
                self.n_cancel_log = 0
                self.n_expiration_log = 0
                self.n_execution_log = 0
                self.n_simulation_begin_log = 0
                self.n_simulation_end_log = 0
                self.n_session_begin_log = 0
                self.n_session_end_log = 0
                self.n_market_step_begin_log = 0
                self.n_market_step_end_log = 0

            def process_order_log(self, log: OrderLog) -> None:
                self.n_order_log += 1

            def process_cancel_log(self, log: CancelLog) -> None:
                self.n_cancel_log += 1

            def process_expiration_log(self, log: ExpirationLog) -> None:
                self.n_expiration_log += 1

            def process_execution_log(self, log: ExecutionLog) -> None:
                self.n_execution_log += 1

            def process_simulation_begin_log(self, log: SimulationBeginLog) -> None:
                self.n_simulation_begin_log += 1

            def process_simulation_end_log(self, log: SimulationEndLog) -> None:
                self.n_simulation_end_log += 1

            def process_session_begin_log(self, log: SessionBeginLog) -> None:
                self.n_session_begin_log += 1

            def process_session_end_log(self, log: SessionEndLog) -> None:
                self.n_session_end_log += 1

            def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
                self.n_market_step_begin_log += 1

            def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
                self.n_market_step_end_log += 1

        logger = DummyLogger()
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        market = Market(
            market_id=2, prng=random.Random(22), simulator=sim, name="test_market"
        )
        order_log = OrderLog(
            order_id=1,
            market_id=2,
            time=3,
            agent_id=4,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=5,
            price=6.0,
            ttl=7,
        )
        logger.write(log=order_log)
        cancel_log = CancelLog(
            order_id=1,
            market_id=2,
            cancel_time=4,
            order_time=3,
            agent_id=5,
            is_buy=False,
            kind=MARKET_ORDER,
            volume=8,
            ttl=9,
        )
        logger.write(log=cancel_log)
        expiration_log = ExpirationLog(
            order_id=1,
            market_id=2,
            time=5,
            order_time=3,
            agent_id=4,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=6,
            price=7.0,
            ttl=2,
        )
        logger.write(log=expiration_log)
        execution_log = ExecutionLog(
            market_id=1,
            time=2,
            buy_agent_id=3,
            sell_agent_id=4,
            buy_order_id=5,
            sell_order_id=6,
            price=7.0,
            volume=8,
        )
        logger.write(log=execution_log)
        simulation_begin_log = SimulationBeginLog(simulator=sim)
        logger.write(log=simulation_begin_log)
        simulation_end_log = SimulationEndLog(simulator=sim)
        logger.write(log=simulation_end_log)
        session_begin_log = SessionBeginLog(session=session, simulator=sim)
        logger.write(log=session_begin_log)
        session_end_log = SessionEndLog(session=session, simulator=sim)
        logger.write(log=session_end_log)
        market_step_begin_log = MarketStepBeginLog(
            session=session, market=market, simulator=sim
        )
        logger.write(log=market_step_begin_log)
        market_step_end_log = MarketStepEndLog(
            session=session, market=market, simulator=sim
        )
        logger.write(log=market_step_end_log)
        logger._process()

        unknown_log = Log()
        with pytest.raises(NotImplementedError):
            logger.write_and_direct_process(log=unknown_log)

        assert logger.n_order_log == 1
        assert logger.n_cancel_log == 1
        assert logger.n_expiration_log == 1
        assert logger.n_execution_log == 1
        assert logger.n_simulation_begin_log == 1
        assert logger.n_simulation_end_log == 1
        assert logger.n_session_begin_log == 1
        assert logger.n_session_end_log == 1
        assert logger.n_market_step_begin_log == 1
        assert logger.n_market_step_end_log == 1

    def test_process2(self) -> None:

        logger = Logger()
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        market = Market(
            market_id=2, prng=random.Random(22), simulator=sim, name="test_market"
        )
        order_log = OrderLog(
            order_id=1,
            market_id=2,
            time=3,
            agent_id=4,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=5,
            price=6.0,
            ttl=7,
        )
        logger.write(log=order_log)
        cancel_log = CancelLog(
            order_id=1,
            market_id=2,
            cancel_time=4,
            order_time=3,
            agent_id=5,
            is_buy=False,
            kind=MARKET_ORDER,
            volume=8,
            ttl=9,
        )
        logger.write(log=cancel_log)
        expiration_log = ExpirationLog(
            order_id=1,
            market_id=2,
            time=5,
            order_time=3,
            agent_id=4,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=6,
            price=7.0,
            ttl=2,
        )
        logger.write(log=expiration_log)
        execution_log = ExecutionLog(
            market_id=1,
            time=2,
            buy_agent_id=3,
            sell_agent_id=4,
            buy_order_id=5,
            sell_order_id=6,
            price=7.0,
            volume=8,
        )
        logger.write(log=execution_log)
        simulation_begin_log = SimulationBeginLog(simulator=sim)
        logger.write(log=simulation_begin_log)
        simulation_end_log = SimulationEndLog(simulator=sim)
        logger.write(log=simulation_end_log)
        session_begin_log = SessionBeginLog(session=session, simulator=sim)
        logger.write(log=session_begin_log)
        session_end_log = SessionEndLog(session=session, simulator=sim)
        logger.write(log=session_end_log)
        market_step_begin_log = MarketStepBeginLog(
            session=session, market=market, simulator=sim
        )
        logger.write(log=market_step_begin_log)
        market_step_end_log = MarketStepEndLog(
            session=session, market=market, simulator=sim
        )
        logger.write(log=market_step_end_log)
        logger._process()

        unknown_log = Log()
        with pytest.raises(NotImplementedError):
            logger.write_and_direct_process(log=unknown_log)
