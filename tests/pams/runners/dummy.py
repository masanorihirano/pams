import time
from typing import Any
from typing import List
from typing import Union

from pams import LIMIT_ORDER
from pams.agents import Agent
from pams.agents import HighFrequencyAgent
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


class GivenOrdersAgent(Agent):
    """Agent that submits the orders set to ``orders_to_submit``.

    This agent is used to check the orders for markets that the agent cannot access.
    ``orders_to_submit`` is set on the main process and copied to worker processes, so that
    this agent works on :class:`pams.runners.MultiProcessAgentParallelRunner` as well.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the orders to submit (none by default)."""
        super().__init__(*args, **kwargs)
        self.orders_to_submit: List[Union[Order, Cancel]] = []

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        return list(self.orders_to_submit)


class HighFrequencyGivenOrdersAgent(GivenOrdersAgent, HighFrequencyAgent):
    """High frequency version of :class:`GivenOrdersAgent`."""
