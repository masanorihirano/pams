import random
import sys
import threading
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pams import LIMIT_ORDER
from pams import Simulator
from pams.agents import Agent
from pams.agents.fcn_agent import FCNAgent
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import MarketStepBeginLog
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
from pams.logs import SessionBeginLog
from pams.logs import SessionEndLog
from pams.logs import SimulationBeginLog
from pams.logs import SimulationEndLog
from pams.market import Market
from pams.order import Cancel
from pams.order import Order

WAIT_TIME = 0.2  # seconds


class FCNDelayAgent(FCNAgent):
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        time.sleep(WAIT_TIME)  # Simulate a delay
        return super().submit_orders(markets)


class DummyLogger(Logger):
    def __init__(self) -> None:
        """Initialize counters."""
        super().__init__()
        self.n_market_step_begin = 0
        self.n_market_end_begin = 0

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        self.n_market_step_begin += 1

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        self.n_market_end_begin += 1


class DummyLogger2(Logger):
    def __init__(self) -> None:
        """Initialize counters."""
        super().__init__()
        self.n_order_log = 0
        self.n_cancel_log = 0
        self.n_execution_log = 0
        self.n_simulation_begin_log = 0
        self.n_simulation_end_log = 0
        self.n_session_begin_log = 0
        self.n_session_end_log = 0
        self.n_market_step_begin = 0
        self.n_market_step_end = 0

    def process_order_log(self, log: OrderLog) -> None:
        self.n_order_log += 1

    def process_cancel_log(self, log: CancelLog) -> None:
        self.n_cancel_log += 1

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
        self.n_market_step_begin += 1

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        self.n_market_step_end += 1


class ExecutionCountLogger(Logger):
    def __init__(self) -> None:
        """Initialize counters."""
        super().__init__()
        self.execution_logs: List[ExecutionLog] = []

    def process_execution_log(self, log: ExecutionLog) -> None:
        self.execution_logs.append(log)


class SimulatorAccessingLogger(Logger):
    def __init__(self) -> None:
        """Initialize counters."""
        super().__init__()
        self.accessed_simulators: List = []

    def process_simulation_begin_log(self, log: SimulationBeginLog) -> None:
        assert self.simulator is log.simulator
        self.accessed_simulators.append(self.simulator)

    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        assert self.simulator is log.simulator
        self.accessed_simulators.append(self.simulator)


class RandomlyIdleFCNAgent(FCNAgent):
    """FCNAgent that submits no orders with 50% probability.

    This agent is used to check that the runners ask the same agents to submit orders
    even when some agents submit no orders.
    """

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        if self.prng.random() < 0.5:
            return []
        return super().submit_orders(markets)


class IdleEvenIDFCNAgent(FCNAgent):
    """FCNAgent that submits no orders if its agent_id is even."""

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        if self.agent_id % 2 == 0:
            return []
        return super().submit_orders(markets)


class CancelingAgent(Agent):
    """Agent that alternately submits a limit buy order and cancels it.

    The state is derived from the order books so that this agent works on
    :class:`pams.runners.MultiProcessAgentParallelRunner` as well.
    """

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            if not self.is_market_accessible(market_id=market.market_id):
                continue
            my_orders = [
                order
                for order in market.buy_order_book.priority_queue
                if order.agent_id == self.agent_id and not order.is_canceled
            ]
            if len(my_orders) > 0:
                orders.append(Cancel(order=my_orders[0]))
            else:
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=1,
                        price=market.get_market_price()
                        * (1 - 0.01 * self.prng.random()),
                        ttl=None,
                    )
                )
        return orders


class RaisingAgent(Agent):
    """Agent that raises an error in submit_orders."""

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        raise RuntimeError("error in submit_orders")


# The functions below are defined at the top level so that the agent-parallel runners can pickle
# them and call them on worker processes.

# token given to the initializer of the current worker thread (or the main thread of a worker
# process). Thread-local storage is used because thread ids can be reused after threads end.
WORKER_STATE = threading.local()
# overwritten only on the main process by the tests; worker processes see it only when forked
PARENT_MARKER: Optional[str] = None


def initialize_worker(token: str) -> None:
    """Record that the current worker is initialized with the token."""
    WORKER_STATE.token = token


def get_worker_token() -> Optional[str]:
    """Get the token of the current worker, or None if it is not initialized."""
    return getattr(WORKER_STATE, "token", None)


def fail_to_initialize_worker() -> None:
    """Raise an error as a worker initializer."""
    raise RuntimeError("error in worker initializer")


def fail_to_initialize_worker_with_system_exit() -> None:
    """Exit as a worker initializer, i.e., raise SystemExit."""
    sys.exit("error in worker initializer")


class WorkerInitializerAbort(BaseException):
    """BaseException that is not an Exception, raised by a worker initializer."""


def fail_to_initialize_worker_with_base_exception() -> None:
    """Raise a BaseException that is not an Exception as a worker initializer."""
    raise WorkerInitializerAbort("error in worker initializer")


def get_parent_marker() -> Optional[str]:
    """Get PARENT_MARKER seen by the current worker."""
    return PARENT_MARKER


class WorkerInitializationCheckingAgent(FCNAgent):
    """FCNAgent that raises an error if the current worker is not initialized with its token.

    The token is given by ``workerToken`` in the settings.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent without the token, which is set by setup."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.worker_token: Optional[str] = None

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.worker_token = settings["workerToken"]

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        worker_token = get_worker_token()
        if worker_token != self.worker_token:
            raise RuntimeError(
                f"worker is initialized with {worker_token}, not {self.worker_token}"
            )
        return super().submit_orders(markets)
