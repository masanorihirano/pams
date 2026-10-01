import random
import sys
import threading
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

from pams import LIMIT_ORDER
from pams import Simulator
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
from pams.session import Session
from pams.transaction_fees import TransactionFee

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


class TransactionFeeRevenueLogger(ExecutionCountLogger):
    """Logger that reads the transaction fee revenues of the markets at the end of each step."""

    def __init__(self) -> None:
        """Initialize the records."""
        super().__init__()
        # (market ID, time, revenue of the time step, cumulative revenue)
        self.revenues: List[Tuple[int, int, float, float]] = []

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """Record the transaction fee revenues of the market in this step."""
        market = log.market
        self.revenues.append(
            (
                market.market_id,
                market.get_time(),
                market.get_transaction_fee_revenue(),
                market.get_cumulative_transaction_fee_revenue(),
            )
        )


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


class MakerTakerTransactionFee(TransactionFee):
    """Transaction fee with a maker rate and a taker rate.

    The order placed first is the maker, and the other is the taker. Order IDs increase with the
    arrival of orders in each market, so they break the tie between orders placed at the same time.
    This class is defined here so that it can be pickled with the markets and used on worker
    processes.
    """

    def __init__(self, market: Market) -> None:
        """Initialize the rates."""
        super().__init__(market=market)
        self.maker_rate: float = 0.0
        self.taker_rate: float = 0.0

    def setup(self, settings: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.maker_rate = float(settings["makerRate"])
        self.taker_rate = float(settings["takerRate"])

    def compute_fees(
        self, price: float, volume: int, buy_order: Order, sell_order: Order
    ) -> Tuple[float, float]:
        value = price * volume
        maker_fee = self.maker_rate * value
        taker_fee = self.taker_rate * value
        buy_arrival = (cast(int, buy_order.placed_at), cast(int, buy_order.order_id))
        sell_arrival = (cast(int, sell_order.placed_at), cast(int, sell_order.order_id))
        if buy_arrival < sell_arrival:
            return maker_fee, taker_fee
        return taker_fee, maker_fee


class GivenOrdersAgent(Agent):
    """Agent that submits the orders set to ``orders_to_submit``.

    This agent is used to check the orders that the runners reject, e.g., the orders for markets
    that the agent cannot access and the cancel orders of the orders of other agents.
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


class LearningAgent(Agent):
    """Agent whose orders depend on the attributes it changes in submit_orders.

    The attributes are listed in ``synced_attributes`` so that
    :class:`pams.runners.MultiProcessAgentParallelRunner` gives the same results as
    :class:`pams.runners.SequentialRunner`. ``n_calls`` is a counter, ``prices`` is a list that
    the agent appends to, ``last_price`` is reassigned and reset to None every fifth call,
    ``bias`` is created lazily and deleted every fourth call, and ``weights`` and
    ``same_weights`` refer to the same list. Every third call submits no orders.
    ``syncedAttributes`` in the settings, if given, replaces ``synced_attributes`` of the
    instance.
    """

    synced_attributes: Union[Tuple[str, ...], List[str]] = (
        "n_calls",
        "prices",
        "last_price",
        "bias",
        "weights",
        "same_weights",
    )

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent with the attributes that it changes in submit_orders."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.n_calls: int = 0
        self.prices: List[float] = []
        self.last_price: Optional[float] = None
        self.weights: List[float] = [1.0]
        self.same_weights: List[float] = self.weights

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        if "syncedAttributes" in settings:
            self.synced_attributes = settings["syncedAttributes"]

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        self.n_calls += 1
        market = next(
            market for market in markets if self.is_market_accessible(market.market_id)
        )
        price = market.get_market_price()
        self.prices.append(price)
        previous_price = price if self.last_price is None else self.last_price
        # a falsy value must be sent back as it is
        self.last_price = None if self.n_calls % 5 == 0 else price
        if not hasattr(self, "bias"):
            # created lazily so that a listed attribute is missing on some calls
            self.bias: float = 0.0  # pylint: disable=attribute-defined-outside-init
        self.bias += 0.01 * (self.prng.random() - 0.5)
        bias = self.bias
        if self.n_calls % 4 == 0:
            del self.bias
        # appended through one name and read through the other
        self.same_weights.append(self.prng.random())
        if self.n_calls % 3 == 0:
            return []
        mean_weight = sum(self.weights) / len(self.weights)
        mean_price = sum(self.prices) / len(self.prices)
        is_buy = price < previous_price or (
            price == previous_price and self.prng.random() < 0.5
        )
        margin = 0.01 * self.prng.random()
        order_price = (
            mean_price
            * (1.0 + bias + 0.01 * (mean_weight - 0.5))
            * (1.0 + margin if is_buy else 1.0 - margin)
        )
        return [
            Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=1,
                price=order_price,
                ttl=None,
            )
        ]


class UnsyncedLearningAgent(LearningAgent):
    """LearningAgent whose class lists no attributes. Tests set them on the class at runtime."""

    synced_attributes = ()


class SlowLearningAgent(LearningAgent):
    """LearningAgent that takes ``WAIT_TIME`` seconds in submit_orders."""

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        time.sleep(WAIT_TIME)
        return super().submit_orders(markets)


class LockingLearningAgent(LearningAgent):
    """LearningAgent that keeps a lock, which cannot be pickled, in ``lock`` in submit_orders."""

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        self.lock = threading.Lock()  # pylint: disable=attribute-defined-outside-init
        return super().submit_orders(markets)


class AgentHelper:
    """Object that keeps the state of its agent and refers back to the agent."""

    def __init__(self, agent: Agent, prices: Optional[List[float]] = None) -> None:
        """Initialize the helper of the agent with the prices it has kept."""
        self.agent: Agent = agent
        self.prices: List[float] = [] if prices is None else prices


class HelperAgent(Agent):
    """Agent that keeps its state in a helper referring back to the agent.

    The helper is listed in ``synced_attributes``. It is replaced with a new one every fourth
    call.
    """

    synced_attributes = ("helper",)

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent with its helper."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.helper: AgentHelper = AgentHelper(agent=self)

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        assert self.helper.agent is self, "the helper does not refer to this agent"
        market = next(
            market for market in markets if self.is_market_accessible(market.market_id)
        )
        price = market.get_market_price()
        prices = self.helper.prices
        prices.append(price)
        if len(prices) % 4 == 0:
            self.helper = AgentHelper(agent=self, prices=prices[-2:])
        mean_price = sum(prices) / len(prices)
        is_buy = price < mean_price or (
            price == mean_price and self.prng.random() < 0.5
        )
        margin = 0.01 * self.prng.random()
        return [
            Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=1,
                price=mean_price * (1.0 + margin if is_buy else 1.0 - margin),
                ttl=None,
            )
        ]


class MarketReferencingAgent(Agent):
    """Agent that keeps the markets and the session in its listed attributes.

    ``last_prices`` is a dict keyed by the markets, and ``last_market`` and ``last_session`` are
    the market and the session of the last call.
    """

    synced_attributes = ("last_prices", "last_market", "last_session")

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent without markets."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.last_prices: Dict[Market, Optional[float]] = {}
        self.last_market: Optional[Market] = None
        self.last_session: Optional[Session] = None

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.last_prices = {
            self.simulator.id2market[market_id]: None
            for market_id in accessible_markets_ids
        }

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        market = next(
            market for market in markets if self.is_market_accessible(market.market_id)
        )
        # raises a KeyError if the keys are not the markets given to this method
        last_price = self.last_prices[market]
        price = market.get_market_price()
        self.last_prices[market] = price
        self.last_market = market
        self.last_session = self.simulator.current_session
        is_buy = (
            self.prng.random() < 0.5
            if last_price is None or last_price == price
            else price < last_price
        )
        margin = 0.01 * self.prng.random()
        return [
            Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=1,
                price=price * (1.0 + margin if is_buy else 1.0 - margin),
                ttl=None,
            )
        ]


class OrderTrackingAgent(Agent):
    """Agent that keeps the orders that it submits in a listed list.

    Each call submits a limit order that expires after ``ORDER_TTL`` steps, and cancels the
    oldest order of the agent that is still in the order books if there are two or more. The
    orders in the order books are found by identity, so they must be the same objects as those
    in ``my_orders``. The side of the order depends on the number of the orders that are fully
    executed, which are no longer in the order books.
    """

    synced_attributes = ("my_orders",)
    ORDER_TTL = 6

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent with no orders."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.my_orders: List[Order] = []

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        market = next(
            market for market in markets if self.is_market_accessible(market.market_id)
        )
        placed_orders = (
            market.buy_order_book.priority_queue + market.sell_order_book.priority_queue
        )
        live_orders = [
            order
            for order in self.my_orders
            if any(order is placed_order for placed_order in placed_orders)
        ]
        orders: List[Union[Order, Cancel]] = []
        if len(live_orders) >= 2:
            orders.append(Cancel(order=live_orders[0]))
        n_executed = sum(order.volume == 0 for order in self.my_orders)
        is_buy = (n_executed + (self.prng.random() < 0.5)) % 2 == 0
        margin = 0.01 * self.prng.random()
        order = Order(
            agent_id=self.agent_id,
            market_id=market.market_id,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            volume=1,
            price=market.get_market_price()
            * (1.0 + margin if is_buy else 1.0 - margin),
            ttl=self.ORDER_TTL,
        )
        self.my_orders.append(order)
        orders.append(order)
        return orders
