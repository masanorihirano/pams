import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from io import TextIOWrapper
from multiprocessing import cpu_count
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from pams.logs.base import Logger
from pams.simulator import Simulator

from ..order import Cancel
from ..order import Order
from ..session import Session
from .sequential import SequentialRunner


class MultiThreadAgentParallelRuner(SequentialRunner):
    """Multi Thread Agent Parallel runner class. This is experimental.

    In this runner, only normal agents are parallelized in each steps.
    This means that the number of agents that can be parallelized is limited by MAX_NORMAL_ORDERS.
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
        super().__init__(settings, prng, logger, simulator_class)
        warnings.warn(
            f"{self.__class__.__name__} is experimental. Future changes may occur disruptively."
        )
        self.num_parallel = max(cpu_count() - 1, 1)

    def _setup(self) -> None:
        super()._setup()
        if "numParallel" in self.settings["simulation"]:
            self.num_parallel = self.settings["simulation"]["numParallel"]
        max_notmal_orders = max(
            session.max_normal_orders for session in self.simulator.sessions
        )
        if self.num_parallel > max_notmal_orders:
            warnings.warn(
                f"When {self.__class__.__name__} is used, the maximum number of parallel agents"
                f" is limited by max_normal_orders ({max_notmal_orders}) evne if numParallel"
                f" ({self.num_parallel}) is set to a larger value."
            )
        self.thread_pool = self._parallel_pool_provider(max_workers=self.num_parallel)

    def _collect_orders_from_normal_agents(
        self, session: Session
    ) -> List[List[Union[Order, Cancel]]]:
        """collect orders from normal_agents. (Internal method)
        orders are corrected until the total number of orders reaches max_normal_orders

        Args:
            session (Session): session.

        Returns:
            List[List[Union[Order, Cancel]]]: orders lists.
        """
        agents = self.simulator.normal_frequency_agents
        agents = self._prng.sample(agents, len(agents))
        all_orders: List[List[Union[Order, Cancel]]] = []
        # TODO: currently the original impl is used for order counting.
        # See more in the SequentialRunner class.
        futures = []
        for agent in agents[: session.max_normal_orders]:
            future = self.thread_pool.submit(
                agent.submit_orders, self.simulator.markets
            )
            futures.append((future, agent))
        for future, agent in futures:
            orders = future.result()
            if len(orders) > 0:
                if not session.with_order_placement:
                    raise AssertionError("currently order is not accepted")
                if sum([order.agent_id != agent.agent_id for order in orders]) > 0:
                    raise ValueError(
                        "spoofing order is not allowed. please check agent_id in order"
                    )
                all_orders.append(orders)
        return all_orders


class MultiProcessAgentParallelRuner(MultiThreadAgentParallelRuner):
    """Multi Process Agent Parallel runner class. This is experimental.

    In this runner, only normal agents are parallelized in each steps.
    This means that the number of agents that can be parallelized is limited by MAX_NORMAL_ORDERS.
    """

    _parallel_pool_provider: Union[
        Type[ThreadPoolExecutor], Type[ProcessPoolExecutor]
    ] = ProcessPoolExecutor
