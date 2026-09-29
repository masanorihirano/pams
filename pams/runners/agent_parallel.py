import multiprocessing
import os
import random
import threading
import warnings
from concurrent.futures import Executor
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from io import TextIOWrapper
from multiprocessing.context import BaseContext
from typing import Any
from typing import Callable
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

# state of the current worker thread or process, set by _initialize_worker
_worker_state = threading.local()


def _initialize_worker(
    initializer: Optional[Callable[..., Any]], initargs: Tuple[Any, ...]
) -> None:
    """Call the worker initializer on a worker (internal function).

    If the initializer raises an exception, the exception is kept and
    :func:`_submit_orders_in_worker` raises an error on this worker instead of breaking the
    executor. This is because a broken :class:`concurrent.futures.ProcessPoolExecutor` can hang
    on Python 3.10 or earlier when large tasks, such as the pickled simulation, are waiting to be
    sent to the worker processes.

    Args:
        initializer (Callable[..., Any], Optional): worker initializer. If it is None, nothing is
            called.
        initargs (Tuple[Any, ...]): arguments of the worker initializer.

    Returns:
        None

    """
    _worker_state.initializer_error = None
    if initializer is not None:
        try:
            initializer(*initargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # any error of the user-defined initializer is raised by the tasks on this worker
            _worker_state.initializer_error = e


def _submit_orders_in_worker(
    agents: List[Agent], markets: List[Market]
) -> List[Tuple[List[Union[Order, Cancel]], Any]]:
    """Call :func:`pams.agents.Agent.submit_orders` of agents on a worker (internal function).

    This function is a module-level function so that it can be pickled and executed
    on a worker process of :class:`concurrent.futures.ProcessPoolExecutor`.
    The agents are asked one by one in the given order.
    If the worker initializer failed on this worker, a RuntimeError is raised with the exception
    raised by the initializer as its cause.

    Args:
        agents (List[Agent]): agents.
        markets (List[Market]): markets.

    Returns:
        List[Tuple[List[Union[Order, Cancel]], Any]]: for each agent, orders submitted by the agent
            and the state of the agent's pseudo random number generator after the submission. The
            state is required to update the agent on the main process when this function runs on
            another process.

    """
    initializer_error: Optional[Exception] = getattr(
        _worker_state, "initializer_error", None
    )
    if initializer_error is not None:
        raise RuntimeError(
            "the worker initializer failed on this worker"
        ) from initializer_error
    results: List[Tuple[List[Union[Order, Cancel]], Any]] = []
    for agent in agents:
        orders: List[Union[Order, Cancel]] = agent.submit_orders(markets=markets)
        results.append((orders, agent.prng.getstate()))
    return results


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

    Subclasses can customize the workers by overriding the following internal methods:

    - ``_create_executor``: creates the executor.
    - ``_get_worker_initializer`` and ``_get_worker_initargs``: a function called once on each
      worker before it runs any task, and its arguments.
    - ``_split_agents_into_chunks``: splits the agents asked in parallel into the tasks of the
      executor.

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
        self.executor = self._create_executor()

    def _create_executor(self) -> Executor:
        """Create a new executor (internal method).

        This method is called by ``_setup`` and ``_get_executor``. Subclasses can override it to
        customize the executor. This runner creates ``_parallel_pool_provider``, i.e.,
        :class:`concurrent.futures.ThreadPoolExecutor`, with ``num_parallel`` workers and
        ``_initialize_worker`` as the initializer, which calls the initializer given by
        ``_get_worker_initializer`` and ``_get_worker_initargs``.

        Returns:
            Executor: new executor.

        """
        return self._parallel_pool_provider(
            max_workers=self.num_parallel,
            initializer=_initialize_worker,
            initargs=(self._get_worker_initializer(), self._get_worker_initargs()),
        )

    def _get_worker_initializer(self) -> Optional[Callable[..., Any]]:
        """Get the function called once on each worker before it runs any task (internal method).

        Subclasses can override this method to set up each worker, e.g., to configure a library
        or to load a model once per worker instead of once per task.
        The function is called with the arguments returned by ``_get_worker_initargs`` on each
        worker thread of :class:`pams.runners.MultiThreadAgentParallelRunner` and on each worker
        process of :class:`pams.runners.MultiProcessAgentParallelRunner`, but never on the main
        thread. For the latter, the function and its arguments are pickled, so the function must be
        defined at the top level of a module that the worker processes can import.
        If the function raises an exception on a worker, every task on the worker raises a
        RuntimeError whose cause is the exception, and the simulation fails with it.

        Returns:
            Callable[..., Any], Optional: initializer of the workers. The default is None, i.e.,
            nothing is called.

        """
        return None

    def _get_worker_initargs(self) -> Tuple[Any, ...]:
        """Get the arguments of the worker initializer (internal method).

        Returns:
            Tuple[Any, ...]: arguments passed to the function returned by
            ``_get_worker_initializer``. The default is an empty tuple.

        """
        return ()

    def _get_executor(self) -> Executor:
        """Get the executor (internal method).

        If the executor is not prepared, prepare it by ``_create_executor``.

        Returns:
            Executor: executor.

        """
        if self.executor is None:
            self.executor = self._create_executor()
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

    def _split_agents_into_chunks(self, agents: List[Agent]) -> List[List[Agent]]:
        """Split the agents asked in parallel into chunks (internal method).

        Each chunk is submitted to the executor as one task, and the agents in a chunk are asked
        one by one on the same worker. This runner makes one chunk per agent so that the agents are
        distributed to idle workers as soon as possible.

        Args:
            agents (List[Agent]): agents asked in parallel.

        Returns:
            List[List[Agent]]: chunks. Their concatenation must be ``agents`` in the same order so
            that the results are the same as :class:`pams.runners.SequentialRunner`; otherwise,
            a ValueError is raised.

        """
        return [[agent] for agent in agents]

    def _receive_orders_from_worker(
        self, agent: Agent, orders: List[Union[Order, Cancel]], prng_state: Any
    ) -> List[Union[Order, Cancel]]:
        """Receive the result of the worker and update the agent (internal method).

        Args:
            agent (Agent): agent on the main process.
            orders (List[Union[Order, Cancel]]): orders submitted by the agent on the worker.
            prng_state (Any): state of the agent's pseudo random number generator on the worker.

        Returns:
            List[Union[Order, Cancel]]: orders submitted by the agent.

        """
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
        Each batch is split into chunks by ``_split_agents_into_chunks``, and each chunk is
        submitted to the executor as one task.

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
            chunks: List[List[Agent]] = self._split_agents_into_chunks(agents=batch)
            chunked_agents: List[Agent] = [agent for chunk in chunks for agent in chunk]
            if len(chunked_agents) != len(batch) or any(
                chunked_agent is not agent
                for chunked_agent, agent in zip(chunked_agents, batch)
            ):
                raise ValueError(
                    "the concatenation of the chunks returned by _split_agents_into_chunks"
                    " must be the given agents in the same order"
                )
            futures: List["Future[List[Tuple[List[Union[Order, Cancel]], Any]]]"] = [
                executor.submit(_submit_orders_in_worker, chunk, markets)
                for chunk in chunks
            ]
            try:
                for chunk, future in zip(chunks, futures):
                    for agent, (orders, prng_state) in zip(chunk, future.result()):
                        orders = self._receive_orders_from_worker(
                            agent=agent, orders=orders, prng_state=prng_state
                        )
                        if len(orders) == 0:
                            continue
                        if not session.with_order_placement:
                            raise AssertionError("currently order is not accepted")
                        if (
                            sum(order.agent_id != agent.agent_id for order in orders)
                            > 0
                        ):
                            raise ValueError(
                                "spoofing order is not allowed. please check agent_id in order"
                            )
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

    The start method of :mod:`multiprocessing` for the worker processes can be set by
    ``simulation.startMethod`` in the config, e.g., ``"spawn"``. It must be one of
    :func:`multiprocessing.get_all_start_methods` on the platform. If it is not set,
    :attr:`default_start_method` is used; if that is None, the default start method of the
    platform is used.

    In addition to the internal methods listed in
    :class:`pams.runners.MultiThreadAgentParallelRunner`, subclasses can customize the worker
    processes by overriding the following members:

    - :attr:`default_start_method`: the start method used when ``simulation.startMethod`` is not
      set. For example, ``"spawn"`` is required for libraries that are not fork-safe, such as
      PyTorch with CUDA, TensorFlow, and JAX.
    - ``_get_mp_context``: the multiprocessing context of the worker processes.

    .. note::
        The agents and the markets are pickled and copied to a worker process for each task.
        Therefore, only the returned orders and the state
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
        of the cost of pickling. Agents and markets refer to the simulator, so every task pickles
        and unpickles the whole simulation: all the agents, all the markets with their histories,
        the events, and the logger. For example, with 1000 :class:`pams.agents.FCNAgent` agents,
        the pickled data is about 4 MB, mostly the states of the agents' pseudo random number
        generators, and it takes tens of milliseconds to pickle and as long to unpickle. To reduce
        the cost, the agents asked in parallel are split into at most ``numParallel`` chunks of
        consecutive agents, and each chunk is one task. Thus, the simulation is pickled at most
        ``numParallel`` times per batch instead of once per agent.
        Large objects held by agents, such as neural network models, are pickled in every task,
        even in the tasks of other agents, unless they are excluded from pickling, e.g., by
        ``__getstate__``. Read-only objects of this kind can be loaded once per worker process by
        the worker initializer (see ``_get_worker_initializer``) instead.
        If you want to use parallelization, it is recommended to use
        :class:`pams.runners.MultiThreadAgentParallelRunner`.
    """

    _parallel_pool_provider: Type[ProcessPoolExecutor] = ProcessPoolExecutor

    #: Optional[str]: start method of the worker processes used when ``simulation.startMethod`` is
    #: not set in the config. None means the default start method of the platform. Subclasses can
    #: override it, e.g., with ``"spawn"`` for libraries that are not fork-safe.
    default_start_method: Optional[str] = None

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
        self.start_method: Optional[str] = self.default_start_method

    def _setup(self) -> None:
        """Set up the simulation (internal method).

        In addition to :func:`pams.runners.MultiThreadAgentParallelRunner._setup`,
        ``simulation.startMethod`` is read before the executor is prepared.
        """
        if (
            "simulation" in self.settings
            and "startMethod" in self.settings["simulation"]
        ):
            start_method = self.settings["simulation"]["startMethod"]
            all_start_methods: List[str] = multiprocessing.get_all_start_methods()
            if (
                not isinstance(start_method, str)
                or start_method not in all_start_methods
            ):
                raise ValueError(
                    f"simulation.startMethod must be one of {all_start_methods}, "
                    f"but {start_method} is given"
                )
            self.start_method = start_method
        super()._setup()

    def _get_mp_context(self) -> BaseContext:
        """Get the multiprocessing context of the worker processes (internal method).

        Subclasses can override this method to use another context, e.g., that of a library
        extending :mod:`multiprocessing`.

        Returns:
            BaseContext: context of ``start_method``. If ``start_method`` is None, the default
            context of the platform is returned.

        """
        return multiprocessing.get_context(self.start_method)

    def _create_executor(self) -> Executor:
        """Create a new executor (internal method).

        This runner creates ``_parallel_pool_provider``, i.e.,
        :class:`concurrent.futures.ProcessPoolExecutor`, with ``num_parallel`` worker processes
        started by the context given by ``_get_mp_context`` and ``_initialize_worker`` as the
        initializer, which calls the initializer given by ``_get_worker_initializer`` and
        ``_get_worker_initargs``.

        Returns:
            Executor: new executor.

        """
        return self._parallel_pool_provider(
            max_workers=self.num_parallel,
            mp_context=self._get_mp_context(),
            initializer=_initialize_worker,
            initargs=(self._get_worker_initializer(), self._get_worker_initargs()),
        )

    def _split_agents_into_chunks(self, agents: List[Agent]) -> List[List[Agent]]:
        """Split the agents asked in parallel into chunks (internal method).

        Each chunk is pickled together with the markets, i.e., with the whole simulation, and sent to
        a worker process as one task. To pickle the simulation as few times as possible, this runner
        splits the agents into at most ``num_parallel`` chunks of consecutive agents, whose sizes
        differ by at most one.

        Args:
            agents (List[Agent]): agents asked in parallel.

        Returns:
            List[List[Agent]]: chunks. Their concatenation is ``agents``.

        """
        n_chunks: int = min(self.num_parallel, len(agents))
        chunks: List[List[Agent]] = []
        i_start = 0
        for i_chunk in range(n_chunks):
            chunk_size = len(agents) // n_chunks + (
                1 if i_chunk < len(agents) % n_chunks else 0
            )
            chunks.append(agents[i_start : i_start + chunk_size])
            i_start += chunk_size
        return chunks

    def _receive_orders_from_worker(
        self, agent: Agent, orders: List[Union[Order, Cancel]], prng_state: Any
    ) -> List[Union[Order, Cancel]]:
        """Receive the result of the worker and update the agent (internal method).

        In addition to
        :func:`pams.runners.MultiThreadAgentParallelRunner._receive_orders_from_worker`,
        the orders referred by cancel orders are replaced with the orders on the main process
        because
        the orders returned from the worker process are copies of them.

        Args:
            agent (Agent): agent on the main process.
            orders (List[Union[Order, Cancel]]): orders submitted by the agent on the worker.
            prng_state (Any): state of the agent's pseudo random number generator on the worker.

        Returns:
            List[Union[Order, Cancel]]: orders submitted by the agent.

        """
        orders = super()._receive_orders_from_worker(
            agent=agent, orders=orders, prng_state=prng_state
        )
        for order in orders:
            if isinstance(order, Cancel):
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
