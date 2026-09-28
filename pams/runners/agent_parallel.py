import os
import random
import warnings
from concurrent.futures import Executor
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from io import TextIOWrapper
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from ..agents import Agent
from ..logs.base import Logger
from ..market import Market
from ..order import Cancel
from ..order import Order
from ..session import Session
from ..simulator import Simulator
from .sequential import SequentialRunner


def _submit_orders_in_worker(
    agent: Agent, markets: List[Market]
) -> Tuple[List[Union[Order, Cancel]], Any]:
    """Call :func:`pams.agents.Agent.submit_orders` on a worker (internal function).

    This function is a module-level function so that it can be pickled and executed
    on a worker process of :class:`concurrent.futures.ProcessPoolExecutor`.

    Args:
        agent (Agent): agent.
        markets (List[Market]): markets.

    Returns:
        Tuple[List[Union[Order, Cancel]], Any]: orders submitted by the agent and the state of
            the agent's pseudo random number generator after the submission. The state is
            required to update the agent on the main process when this function runs on
            another process.

    """
    orders: List[Union[Order, Cancel]] = agent.submit_orders(markets=markets)
    return orders, agent.prng.getstate()


class MultiThreadAgentParallelRunner(SequentialRunner):
    """Multi Thread Agent Parallel runner class. This is experimental.

    In this runner, only :func:`pams.agents.Agent.submit_orders` of normal (non high-frequency)
    agents
    is parallelized in each step using :class:`concurrent.futures.ThreadPoolExecutor`.
    Order handling, executions, high-frequency agents, events, and logging are processed
    sequentially
    on the main thread in the same way as :class:`pams.runners.SequentialRunner`.

    The simulation results are identical to those of :class:`pams.runners.SequentialRunner` with the
    same
    settings and the same seed, because agents are asked to submit orders in the same order and
    the same number of agents are asked as :class:`pams.runners.SequentialRunner` does.
    Agents are asked in batches: at first, ``maxNormalOrders`` agents are asked in parallel,
    and if some of them submit no orders, the same number of the next agents are asked in parallel,
    and so on, until the number of agents submitting orders reaches ``maxNormalOrders``.
    This means that the number of agents that can be parallelized is limited by ``maxNormalOrders``
    of the session.

    The number of workers can be set by ``simulation.numParallel`` in the config. The default is
    the number of CPUs minus one (at least one).

    .. note::
        :func:`pams.agents.Agent.submit_orders` is called on worker threads concurrently.
        The built-in agents only read markets and use their own pseudo random number generators,
        which is thread-safe in this runner because the main thread waits for all the workers before
        handling orders. User-defined agents must not modify shared objects such as markets,
        the simulator, the logger, or other agents in :func:`pams.agents.Agent.submit_orders`.

    .. note::
        Because of the GIL of python, this runner does not speed up CPU-bound agents such as the
        built-in
        agents. This runner is beneficial when :func:`pams.agents.Agent.submit_orders` waits for I/O
        (e.g., network access to external models or services).
    """

    _parallel_pool_provider: Union[
        Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]
    ] = ThreadPoolExecutor

    def __init__(
        self,
        settings: Union[Dict, TextIOWrapper, os.PathLike, str],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator,
    ):
        """Initialize.

        Args:
            settings (Union[Dict, TextIOWrapper, os.PathLike, str]): runner configuration.
            prng (random.Random, Optional): pseudo random number generator for this runner.
            logger (Logger, Optional): logger instance.
            simulator_class (Type[Simulator]): type of simulator.

        Returns:
            None

        """
        super().__init__(settings, prng, logger, simulator_class)
        warnings.warn(
            f"{self.__class__.__name__} is experimental. Future changes may occur disruptively.",
            stacklevel=2,
        )
        self.num_parallel: int = max((os.cpu_count() or 1) - 1, 1)
        self.executor: Optional[Executor] = None

    def _setup(self) -> None:
        """Set up the simulation (internal method).

        In addition to :func:`pams.runners.SequentialRunner._setup`, ``simulation.numParallel`` is
        read
        and the executor is prepared.
        """
        super()._setup()
        if "numParallel" in self.settings["simulation"]:
            num_parallel = self.settings["simulation"]["numParallel"]
            if (
                isinstance(num_parallel, bool)
                or not isinstance(num_parallel, int)
                or num_parallel < 1
            ):
                raise ValueError(
                    "simulation.numParallel must be a positive integer, "
                    f"but {num_parallel} is given"
                )
            self.num_parallel = num_parallel
        max_normal_orders = max(
            (session.max_normal_orders for session in self.simulator.sessions),
            default=0,
        )
        if self.num_parallel > max_normal_orders:
            warnings.warn(
                f"When {self.__class__.__name__} is used, the maximum number of parallel agents"
                f" is limited by max_normal_orders ({max_normal_orders}) even if numParallel"
                f" ({self.num_parallel}) is set to a larger value.",
                stacklevel=2,
            )
        self._shutdown_executor()
        self.executor = self._parallel_pool_provider(max_workers=self.num_parallel)

    def _get_executor(self) -> Executor:
        """Get the executor (internal method).

        If the executor is not prepared, prepare it.

        Returns:
            Executor: executor.

        """
        if self.executor is None:
            self.executor = self._parallel_pool_provider(max_workers=self.num_parallel)
        return self.executor

    def _shutdown_executor(self) -> None:
        """Shutdown the executor if it exists (internal method)."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def _run(self) -> None:
        """Run the simulation (internal method).

        The executor is shut down after the simulation.
        """
        try:
            super()._run()
        finally:
            self._shutdown_executor()

    def _receive_orders_from_worker(
        self, agent: Agent, future: "Future[Tuple[List[Union[Order, Cancel]], Any]]"
    ) -> List[Union[Order, Cancel]]:
        """Receive the result of the worker and update the agent (internal method).

        Args:
            agent (Agent): agent on the main process.
            future (Future): future of :func:`_submit_orders_in_worker`.

        Returns:
            List[Union[Order, Cancel]]: orders submitted by the agent.

        """
        orders, prng_state = future.result()
        agent.prng.setstate(prng_state)
        return orders

    def _collect_orders_from_normal_agents(
        self, session: Session
    ) -> List[List[Union[Order, Cancel]]]:
        """Collect orders from normal_agents in parallel (internal method).

        Orders are collected until the number of agents submitting orders reaches max_normal_orders.

        Agents are asked in batches so that the agents asked to submit orders and their order are
        exactly the same as
        :func:`pams.runners.SequentialRunner._collect_orders_from_normal_agents`.

        Args:
            session (Session): session.

        Returns:
            List[List[Union[Order, Cancel]]]: orders lists.

        """
        agents = self.simulator.normal_frequency_agents
        agents = self._prng.sample(agents, len(agents))
        markets: List[Market] = self.simulator.markets
        executor: Executor = self._get_executor()
        n_orders = 0
        i_next_agent = 0
        all_orders: List[List[Union[Order, Cancel]]] = []
        while n_orders < session.max_normal_orders and i_next_agent < len(agents):
            # the orders are counted in the same way as SequentialRunner, i.e.,
            # by the number of agents submitting orders.
            n_remaining = session.max_normal_orders - n_orders
            batch: List[Agent] = agents[i_next_agent : i_next_agent + n_remaining]
            i_next_agent += len(batch)
            futures: List["Future[Tuple[List[Union[Order, Cancel]], Any]]"] = [
                executor.submit(_submit_orders_in_worker, agent, markets)
                for agent in batch
            ]
            try:
                for agent, future in zip(batch, futures):
                    orders: List[
                        Union[Order, Cancel]
                    ] = self._receive_orders_from_worker(agent=agent, future=future)
                    if len(orders) > 0:
                        if not session.with_order_placement:
                            raise AssertionError("currently order is not accepted")
                        self._check_submitted_orders(agent=agent, orders=orders)
                        all_orders.append(orders)
                        n_orders += 1
            finally:
                for future in futures:
                    future.cancel()
        return all_orders


class MultiProcessAgentParallelRunner(MultiThreadAgentParallelRunner):
    """Multi Process Agent Parallel runner class. This is experimental.

    In this runner, only :func:`pams.agents.Agent.submit_orders` of normal (non high-frequency)
    agents
    is parallelized in each step using :class:`concurrent.futures.ProcessPoolExecutor`.
    See :class:`pams.runners.MultiThreadAgentParallelRunner` for the details of the parallelization.

    .. note::
        The agent and the markets are pickled and copied to a worker process for each call of
        :func:`pams.agents.Agent.submit_orders`. Therefore, only the returned orders and the state
        of
        the agent's pseudo random number generator are reflected to the agent on the main process.
        Other attributes modified in :func:`pams.agents.Agent.submit_orders` are discarded.
        User-defined agents for this runner should keep their states through the callbacks such as
        :func:`pams.agents.Agent.submitted_order` and :func:`pams.agents.Agent.executed_order`,
        which are
        called on the main process, or derive their states from markets.

    .. note::
        User-defined classes (agents, markets, loggers, etc.) must be picklable. Especially when the
        start method of multiprocessing is ``spawn`` or ``forkserver`` (e.g., macOS and Windows),
        the
        definition of user-defined classes must be importable from worker processes, i.e., they
        should be
        saved to a module file rather than defined in ``__main__`` or in an interactive session.

    .. warning::
        This class is much slower than :class:`pams.runners.MultiThreadAgentParallelRunner` because
        of the
        cost of pickling. If you want to use parallelization, it is recommended to use
        :class:`pams.runners.MultiThreadAgentParallelRunner`.
    """

    _parallel_pool_provider: Union[
        Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]
    ] = ProcessPoolExecutor

    def _receive_orders_from_worker(
        self, agent: Agent, future: "Future[Tuple[List[Union[Order, Cancel]], Any]]"
    ) -> List[Union[Order, Cancel]]:
        """Receive the result of the worker and update the agent (internal method).

        In addition to
        :func:`pams.runners.MultiThreadAgentParallelRunner._receive_orders_from_worker`,
        the orders referred by cancel orders are replaced with the orders on the main process
        because
        the orders returned from the worker process are copies of them.

        Args:
            agent (Agent): agent on the main process.
            future (Future): future of :func:`_submit_orders_in_worker`.

        Returns:
            List[Union[Order, Cancel]]: orders submitted by the agent.

        """
        orders = super()._receive_orders_from_worker(agent=agent, future=future)
        for order in orders:
            if isinstance(order, Cancel):
                if order.order.market_id not in self.simulator.id2market:
                    # rejected later by SequentialRunner._check_submitted_orders
                    continue
                market: Market = self.simulator.id2market[order.order.market_id]
                order_book = (
                    market.buy_order_book
                    if order.order.is_buy
                    else market.sell_order_book
                )
                for placed_order in order_book.priority_queue:
                    if placed_order == order.order:
                        order.order = placed_order
                        break
        return orders
