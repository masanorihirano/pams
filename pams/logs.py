from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

from pams.order import OrderKind


class Logger:
    def __init__(self) -> None:
        self.pending_logs: List[Log] = []

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
        pass


class Log:
    def read_and_write(self, logger: Logger) -> None:
        logger.write(log=self)

    def read_and_write_with_direct_process(self, logger: Logger) -> None:
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


class SimulationEndLog(SimulationBeginLog):
    pass


class SessionBeginLog(Log):
    def __init__(self, session: "Session", simulator: "Simulator"):  # type: ignore
        self.simulator = simulator
        self.session = session


class SessionEndLog(SessionBeginLog):
    pass


class MarketStepBeginLog(Log):
    def __init__(self, market: "Market", simulator: "Simulator"):  # type: ignore
        self.market = market
        self.simulator = simulator


class MarketStepEndLog(MarketStepBeginLog):
    pass
