from typing import List
from typing import Optional

from pams.order import OrderKind


class Log:
    def read_and_write(self, logger: "Logger") -> None:
        logger.write(log=self)

    def read_and_write_with_direct_process(self, logger: "Logger") -> None:
        logger.write_and_direct_process(log=self)


class OrderLog(Log):
    def __init__(
        self,
        order_id: int,
        market_id: int,
        time: int,
        agent_id: int,
        is_buy: bool,
        kind: OrderKind,
        volume: int,
        price: Optional[float] = None,
        ttl: Optional[int] = None,
    ):
        self.order_id: int = order_id
        self.market_id: int = market_id
        self.time: int = time
        self.agent_id: int = agent_id
        self.is_buy: bool = is_buy
        self.kind: OrderKind = kind
        self.price: Optional[float] = price
        self.volume: int = volume
        self.ttl: Optional[int] = ttl


class CancelLog(Log):
    def __init__(
        self,
        order_id: int,
        market_id: int,
        cancel_time: int,
        order_time: int,
        agent_id: int,
        is_buy: bool,
        kind: OrderKind,
        volume: int,
        price: Optional[float] = None,
        ttl: Optional[int] = None,
    ):
        self.order_id: int = order_id
        self.market_id: int = market_id
        self.cancel_time: int = cancel_time
        self.order_time: int = order_time
        self.agent_id: int = agent_id
        self.is_buy: bool = is_buy
        self.kind: OrderKind = kind
        self.price: Optional[float] = price
        self.volume: int = volume
        self.ttl: Optional[int] = ttl


class ExecutionLog(Log):
    def __init__(
        self,
        market_id: int,
        time: int,
        buy_agent_id: int,
        sell_agent_id: int,
        buy_order_id: int,
        sell_order_id: int,
        price: float,
        volume: int,
    ):
        self.market_id: int = market_id
        self.time: int = time
        self.buy_agent_id: int = buy_agent_id
        self.sell_agent_id: int = sell_agent_id
        self.buy_order_id: int = buy_order_id
        self.sell_order_id: int = sell_order_id
        self.price: float = price
        self.volume: int = volume


class SimulationBeginLog(Log):
    def __init__(self, simulator: "Simulator"):  # type: ignore
        self.simulator = simulator


class SimulationEndLog(Log):
    def __init__(self, simulator: "Simulator"):  # type: ignore
        self.simulator = simulator


class SessionBeginLog(Log):
    def __init__(self, session: "Session", simulator: "Simulator"):  # type: ignore
        self.simulator = simulator
        self.session = session


class SessionEndLog(Log):
    def __init__(self, session: "Session", simulator: "Simulator"):  # type: ignore
        self.simulator = simulator
        self.session = session


class MarketStepBeginLog(Log):
    def __init__(self, session: "Session", market: "Market", simulator: "Simulator"):  # type: ignore
        self.session = session
        self.market = market
        self.simulator = simulator


class MarketStepEndLog(Log):
    def __init__(self, session: "Session", market: "Market", simulator: "Simulator"):  # type: ignore
        self.session = session
        self.market = market
        self.simulator = simulator


class Logger:
    simulator: "Simulator"  # type: ignore

    def __init__(self) -> None:
        self.pending_logs: List[Log] = []

    def _set_simulator(self, simulator: "Simulator") -> None:  # type: ignore
        self.simulator = simulator

    def write(self, log: "Log") -> None:
        self.pending_logs.append(log)

    def bulk_write(self, logs: List["Log"]) -> None:
        self.pending_logs.extend(logs)

    def write_and_direct_process(self, log: "Log") -> None:
        self.process(logs=[log])

    def bulk_write_and_direct_process(self, logs: List["Log"]) -> None:
        self.process(logs=logs)

    def _process(self) -> None:
        self.process(logs=self.pending_logs)
        self.pending_logs = []

    def process(self, logs: List["Log"]) -> None:
        for log in logs:
            if isinstance(log, OrderLog):
                self.process_order_log(log=log)
            elif isinstance(log, CancelLog):
                self.process_cancel_log(log=log)
            elif isinstance(log, ExecutionLog):
                self.process_execution_log(log=log)
            elif isinstance(log, SimulationBeginLog):
                self.process_simulation_begin_log(log=log)
            elif isinstance(log, SimulationEndLog):
                self.process_simulation_end_log(log=log)
            elif isinstance(log, SessionBeginLog):
                self.process_session_begin_log(log=log)
            elif isinstance(log, SessionEndLog):
                self.process_session_end_log(log=log)
            elif isinstance(log, MarketStepBeginLog):
                self.process_market_step_begin_log(log=log)
            elif isinstance(log, MarketStepEndLog):
                self.process_market_step_end_log(log=log)
            else:
                raise NotImplementedError

    def process_order_log(self, log: "OrderLog") -> None:
        pass

    def process_cancel_log(self, log: "CancelLog") -> None:
        pass

    def process_execution_log(self, log: "ExecutionLog") -> None:
        pass

    def process_simulation_begin_log(self, log: "SimulationBeginLog") -> None:
        pass

    def process_simulation_end_log(self, log: "SimulationEndLog") -> None:
        pass

    def process_session_begin_log(self, log: "SessionBeginLog") -> None:
        pass

    def process_session_end_log(self, log: "SessionEndLog") -> None:
        pass

    def process_market_step_begin_log(self, log: "MarketStepBeginLog") -> None:
        pass

    def process_market_step_end_log(self, log: "MarketStepEndLog") -> None:
        pass
