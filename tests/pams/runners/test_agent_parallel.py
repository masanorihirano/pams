import copy
import io
import multiprocessing
import os
import pickle
import random
import re
import time
import traceback
import uuid
from concurrent.futures import BrokenExecutor
from concurrent.futures import Executor
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import pytest

from pams import LIMIT_ORDER
from pams import Simulator
from pams.agents import Agent
from pams.order import Cancel
from pams.order import Order
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners import agent_parallel
from pams.runners.agent_parallel import _check_synced_attributes
from pams.runners.agent_parallel import _load_worker_results
from pams.runners.agent_parallel import _SimulationObjects
from pams.runners.agent_parallel import _submit_orders_in_worker
from pams.runners.agent_parallel import _WorkerResultPickler
from pams.runners.sequential import SequentialRunner
from tests.pams.runners.test_sequential import TestSequentialRunner

from . import dummy
from .dummy import WAIT_TIME
from .dummy import CancelingAgent
from .dummy import DummyLogger2
from .dummy import FCNDelayAgent
from .dummy import HelperAgent
from .dummy import IdleEvenIDFCNAgent
from .dummy import LearningAgent
from .dummy import MarketReferencingAgent
from .dummy import OrderTrackingAgent
from .dummy import RaisingAgent
from .dummy import RandomlyIdleFCNAgent
from .dummy import SlowLearningAgent
from .dummy import UnsyncedLearningAgent
from .dummy import WorkerInitializationCheckingAgent
from .dummy import WorkerInitializerAbort
from .dummy import fail_to_initialize_worker
from .dummy import fail_to_initialize_worker_with_base_exception
from .dummy import fail_to_initialize_worker_with_system_exit
from .dummy import get_parent_marker
from .dummy import get_worker_token
from .dummy import initialize_worker


class SpawnMultiProcessAgentParallelRunner(MultiProcessAgentParallelRunner):
    default_start_method = "spawn"


class CustomThreadPoolExecutor(ThreadPoolExecutor):
    pass


class CustomProcessPoolExecutor(ProcessPoolExecutor):
    pass


@pytest.fixture(autouse=True)
def shut_down_executors(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Shut down the executors created by the runners after each test.

    Some tests, e.g., the ones inherited from TestSequentialRunner, leave the executor running.
    When such an executor is garbage-collected, its manager thread shuts it down in the
    background. With the fork start method (the default on Linux before Python 3.14), a worker
    process forked by another executor meanwhile can inherit a lock held by that thread and hang
    when it exits.
    """
    executors: List[Executor] = []

    def track(
        create_executor: Callable[[MultiThreadAgentParallelRunner], Executor]
    ) -> Callable[[MultiThreadAgentParallelRunner], Executor]:
        def create_and_track_executor(
            runner: MultiThreadAgentParallelRunner,
        ) -> Executor:
            executor = create_executor(runner)
            executors.append(executor)
            return executor

        return create_and_track_executor

    for runner_class in [
        MultiThreadAgentParallelRunner,
        MultiProcessAgentParallelRunner,
    ]:
        monkeypatch.setattr(
            runner_class, "_create_executor", track(runner_class._create_executor)
        )
    yield
    for executor in executors:
        executor.shutdown(wait=True)


def _order_keys(orders: List[Union[Order, Cancel]]) -> List[Any]:
    return [
        (order.agent_id, order.order.order_id)
        if isinstance(order, Cancel)
        else (order.agent_id, order.is_buy, order.price, order.volume)
        for order in orders
    ]


def _agent_states(agents: List[Agent]) -> List[Dict[str, Any]]:
    return [
        {
            "agent_id": agent.agent_id,
            "cash_amount": agent.cash_amount,
            "asset_volumes": dict(agent.asset_volumes),
            "prng_state": agent.prng.getstate(),
        }
        for agent in agents
    ]


def _learning_states(agents: List[Agent]) -> List[Dict[str, Any]]:
    return [
        {
            name: getattr(agent, name, "missing")
            for name in LearningAgent.synced_attributes
        }
        for agent in agents
    ]


def _load_results(results: Union[List[Any], bytes], simulator: Simulator) -> List[Any]:
    """Load the results of _submit_orders_in_worker in the same way as the runners."""
    return _load_worker_results(
        results=results, objects=_SimulationObjects(simulator=simulator)
    )


def _placed_orders(simulator: Simulator) -> List[Order]:
    return [
        order
        for market in simulator.markets
        for order_book in [market.buy_order_book, market.sell_order_book]
        for order in order_book.priority_queue
    ]


def _reference_states(agents: List[Agent]) -> List[Any]:
    """Get the states of HelperAgent, MarketReferencingAgent, and OrderTrackingAgent."""
    states: List[Any] = []
    for agent in agents:
        if isinstance(agent, HelperAgent):
            states.append(agent.helper.prices)
        elif isinstance(agent, MarketReferencingAgent):
            states.append(
                (
                    {
                        market.market_id: price
                        for market, price in agent.last_prices.items()
                    },
                    None if agent.last_market is None else agent.last_market.market_id,
                    (
                        None
                        if agent.last_session is None
                        else agent.last_session.session_id
                    ),
                )
            )
        else:
            assert isinstance(agent, OrderTrackingAgent)
            states.append(
                [
                    (
                        order.order_id,
                        order.is_buy,
                        order.price,
                        order.volume,
                        order.placed_at,
                        order.is_canceled,
                    )
                    for order in agent.my_orders
                ]
            )
    return states


def _assert_references_to_simulation(simulator: Simulator) -> None:
    """Check that the listed attributes refer to the objects of the simulation."""
    placed_orders = _placed_orders(simulator=simulator)
    for agent in simulator.normal_frequency_agents:
        if isinstance(agent, HelperAgent):
            assert agent.helper.agent is agent
        elif isinstance(agent, MarketReferencingAgent):
            assert all(
                market is simulator.id2market[market.market_id]
                for market in agent.last_prices
            )
            if agent.last_market is not None:
                assert agent.last_market is simulator.markets[0]
                assert agent.last_session is simulator.sessions[0]
        elif isinstance(agent, OrderTrackingAgent):
            for order in placed_orders:
                if order.agent_id == agent.agent_id:
                    assert any(order is my_order for my_order in agent.my_orders)


def _track_received_results(
    runner: MultiThreadAgentParallelRunner, monkeypatch: pytest.MonkeyPatch
) -> List[Tuple[str, int]]:
    """Record the results of the workers received by the runner.

    Each event is ("attributes", agent_id) for _receive_synced_attributes_from_worker, or
    ("orders", agent_id) or ("no orders", agent_id) for _receive_orders_from_worker.
    """
    received: List[Tuple[str, int]] = []
    receive_synced_attributes = runner._receive_synced_attributes_from_worker
    receive_orders = runner._receive_orders_from_worker

    def receive_synced_attributes_from_worker(
        agent: Agent, attributes: Dict[str, Any]
    ) -> None:
        received.append(("attributes", agent.agent_id))
        receive_synced_attributes(agent=agent, attributes=attributes)

    def receive_orders_from_worker(
        agent: Agent, orders: List[Union[Order, Cancel]], prng_state: Any
    ) -> List[Union[Order, Cancel]]:
        received.append(("orders" if orders else "no orders", agent.agent_id))
        return receive_orders(agent=agent, orders=orders, prng_state=prng_state)

    monkeypatch.setattr(
        runner,
        "_receive_synced_attributes_from_worker",
        receive_synced_attributes_from_worker,
    )
    monkeypatch.setattr(
        runner, "_receive_orders_from_worker", receive_orders_from_worker
    )
    return received


def _assert_same_results(
    sequential_runner: SequentialRunner,
    parallel_runner: SequentialRunner,
    agent_class: str,
) -> None:
    sequential_market = sequential_runner.simulator.markets[0]
    parallel_market = parallel_runner.simulator.markets[0]
    times = range(sequential_market.get_time() + 1)
    assert sequential_market.get_time() == parallel_market.get_time()
    assert sequential_market.get_market_prices(
        times
    ) == parallel_market.get_market_prices(times)
    assert sequential_market.get_fundamental_prices(
        times
    ) == parallel_market.get_fundamental_prices(times)
    assert sequential_market.get_executed_volumes(
        times
    ) == parallel_market.get_executed_volumes(times)
    assert sequential_market.get_n_buy_orders(
        times
    ) == parallel_market.get_n_buy_orders(times)
    assert sequential_market.get_n_sell_orders(
        times
    ) == parallel_market.get_n_sell_orders(times)
    assert _agent_states(sequential_runner.simulator.agents) == _agent_states(
        parallel_runner.simulator.agents
    )
    assert sequential_runner._prng.getstate() == parallel_runner._prng.getstate()
    sequential_logger = sequential_runner.logger
    parallel_logger = parallel_runner.logger
    assert isinstance(sequential_logger, DummyLogger2)
    assert isinstance(parallel_logger, DummyLogger2)
    assert sequential_logger.n_order_log == parallel_logger.n_order_log
    assert sequential_logger.n_cancel_log == parallel_logger.n_cancel_log
    assert sequential_logger.n_execution_log == parallel_logger.n_execution_log
    assert sequential_logger.n_order_log > 0
    if agent_class == "CancelingAgent":
        assert parallel_logger.n_cancel_log > 0
        for market in parallel_runner.simulator.markets:
            for order in market.buy_order_book.priority_queue:
                assert not order.is_canceled


class TestMultiThreadAgentParallelRunner(TestSequentialRunner):
    runner_class: Type[SequentialRunner] = MultiThreadAgentParallelRunner
    default_setting: Dict = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["FCNAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 5,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                    "events": ["FundamentalPriceShock"],
                    "maxNormalOrders": 3,
                }
            ],
            "numParallel": 3,
        },
        "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
        "FCNAgents": {
            "class": "FCNAgent",
            "numAgents": 10,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": {"expon": [1.0]},
            "chartWeight": {"expon": [0.0]},
            "noiseWeight": {"expon": [1.0]},
            "meanReversionTime": {"uniform": [50, 100]},
            "noiseScale": 0.001,
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
        },
        "FundamentalPriceShock": {
            "class": "FundamentalPriceShock",
            "target": "Market",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 1,
            "enabled": True,
        },
    }
    user_classes: List[Type] = [
        FCNDelayAgent,
        RandomlyIdleFCNAgent,
        IdleEvenIDFCNAgent,
        CancelingAgent,
        RaisingAgent,
        WorkerInitializationCheckingAgent,
        LearningAgent,
        UnsyncedLearningAgent,
        SlowLearningAgent,
        HelperAgent,
        MarketReferencingAgent,
        OrderTrackingAgent,
    ]
    custom_pool_provider: Type[Executor] = CustomThreadPoolExecutor
    # whether the runner receives the attributes listed in synced_attributes from the workers
    receives_synced_attributes: bool = False

    def _make_runner(
        self, runner_class: Type[SequentialRunner], setting: Dict, seed: int = 42
    ) -> SequentialRunner:
        runner = runner_class(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        for cls in self.user_classes:
            runner.class_register(cls=cls)
        return runner

    def _make_runners(
        self, setting: Dict, seed: int = 42
    ) -> Tuple[SequentialRunner, MultiThreadAgentParallelRunner]:
        sequential_runner = self._make_runner(
            runner_class=SequentialRunner, setting=setting, seed=seed
        )
        parallel_runner = self._make_runner(
            runner_class=self.runner_class, setting=setting, seed=seed
        )
        assert isinstance(parallel_runner, MultiThreadAgentParallelRunner)
        return sequential_runner, parallel_runner

    @pytest.mark.parametrize(
        "agent_class", ["FCNAgent", "RandomlyIdleFCNAgent", "CancelingAgent"]
    )
    @pytest.mark.parametrize("num_parallel", [1, 2, 3])
    @pytest.mark.parametrize("max_normal_orders", [1, 3, 10, 15])
    def test_same_result_as_sequential(
        self, agent_class: str, num_parallel: int, max_normal_orders: int
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = agent_class
        setting["simulation"]["numParallel"] = num_parallel
        setting["simulation"]["sessions"][0]["iterationSteps"] = 30
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = max_normal_orders
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()

        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class=agent_class,
        )

    def test_collect_orders_from_normal_agents_asks_same_agents(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "IdleEvenIDFCNAgent"
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = 3
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner.simulator._update_times_on_markets(
            sequential_runner.simulator.markets
        )
        parallel_runner.simulator._update_times_on_markets(
            parallel_runner.simulator.markets
        )
        sequential_states_before = _agent_states(sequential_runner.simulator.agents)
        parallel_states_before = _agent_states(parallel_runner.simulator.agents)
        assert sequential_states_before == parallel_states_before

        sequential_orders = sequential_runner._collect_orders_from_normal_agents(
            session=sequential_runner.simulator.sessions[0]
        )
        parallel_orders = parallel_runner._collect_orders_from_normal_agents(
            session=parallel_runner.simulator.sessions[0]
        )
        # only odd agents submit orders; three lists are collected in both runners
        assert len(sequential_orders) == 3
        assert len(parallel_orders) == 3
        assert [
            [order.agent_id for order in orders] for orders in sequential_orders
        ] == [[order.agent_id for order in orders] for orders in parallel_orders]
        assert [_order_keys(orders) for orders in sequential_orders] == [
            _order_keys(orders) for orders in parallel_orders
        ]
        sequential_states_after = _agent_states(sequential_runner.simulator.agents)
        parallel_states_after = _agent_states(parallel_runner.simulator.agents)
        assert sequential_states_after == parallel_states_after
        # the agents asked to submit orders (whose prng advanced) are the same as SequentialRunner
        asked_agents = [
            before["agent_id"]
            for before, after in zip(sequential_states_before, sequential_states_after)
            if before["prng_state"] != after["prng_state"]
        ]
        assert len(asked_agents) >= 3
        parallel_asked_agents = [
            before["agent_id"]
            for before, after in zip(parallel_states_before, parallel_states_after)
            if before["prng_state"] != after["prng_state"]
        ]
        assert asked_agents == parallel_asked_agents
        parallel_runner._shutdown_executor()
        assert parallel_runner.executor is None

    def test_collect_orders_from_normal_agents_max_normal_orders_zero(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = 0
        setting["simulation"]["numParallel"] = 1
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()
        states_before = _agent_states(runner.simulator.agents)
        orders = runner._collect_orders_from_normal_agents(
            session=runner.simulator.sessions[0]
        )
        assert not orders
        assert _agent_states(runner.simulator.agents) == states_before

    def test_exception_in_submit_orders_is_raised(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "RaisingAgent"
        _, runner = self._make_runners(setting=setting)
        with pytest.raises(RuntimeError, match="error in submit_orders"):
            runner.main()
        assert runner.executor is None

    def test_num_parallel_default(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        del setting["simulation"]["numParallel"]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        assert isinstance(runner, MultiThreadAgentParallelRunner)
        assert runner.num_parallel == max((os.cpu_count() or 1) - 1, 1)
        runner._setup()
        assert runner.num_parallel == max((os.cpu_count() or 1) - 1, 1)
        runner._shutdown_executor()

    @pytest.mark.parametrize("num_parallel", [0, -1, 1.5, "2", True, None])
    def test_num_parallel_invalid(self, num_parallel: Any) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["numParallel"] = num_parallel
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError, match="numParallel"):
            runner._setup()

    def test_executor_lifecycle(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        assert isinstance(runner, MultiThreadAgentParallelRunner)
        assert runner.executor is None
        runner._setup()
        executor = runner.executor
        assert executor is not None
        assert isinstance(executor, self.runner_class._parallel_pool_provider)
        assert runner._get_executor() is executor
        # the executor is shut down after the simulation
        runner._run()
        assert runner.executor is None
        with pytest.raises(RuntimeError):
            executor.submit(time.sleep, 0)
        # the executor is lazily created if necessary
        new_executor = runner._get_executor()
        assert new_executor is not executor
        assert runner.executor is new_executor
        assert new_executor.submit(sum, [1, 2]).result() == 3
        runner._shutdown_executor()
        assert runner.executor is None
        with pytest.raises(RuntimeError):
            new_executor.submit(time.sleep, 0)
        # shutdown is idempotent
        runner._shutdown_executor()
        assert runner.executor is None

    def test_parallel_efficiency(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "FCNDelayAgent"  # Use the delayed agent
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        # 5 steps x 3 agents x WAIT_TIME in sequential
        start_time = time.time()
        sequential_runner.main()
        end_time = time.time()
        elps_time_sequential = end_time - start_time
        # 5 steps x 1 batch x WAIT_TIME in parallel
        start_time = time.time()
        parallel_runner.main()
        end_time = time.time()
        elps_time_parallel = end_time - start_time
        overhead_time = max(elps_time_sequential - WAIT_TIME * 15, 0.0)
        assert elps_time_sequential > WAIT_TIME * 15
        assert elps_time_parallel > WAIT_TIME * 5
        assert elps_time_parallel < WAIT_TIME * 5 + overhead_time + 1
        assert elps_time_parallel < elps_time_sequential

    def test_parallel_thread_warning(self) -> None:
        settings = copy.deepcopy(self.default_setting)
        settings["simulation"]["numParallel"] = 5
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=settings
        )
        with pytest.warns(UserWarning, match="is set to a larger value."):
            runner.main()

    def test_experimental_warning(self) -> None:
        with pytest.warns(UserWarning, match="is experimental"):
            self.runner_class(settings=copy.deepcopy(self.default_setting))

    def test_parallel_pool_provider_is_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        assert isinstance(runner, MultiThreadAgentParallelRunner)
        monkeypatch.setattr(
            runner, "_parallel_pool_provider", self.custom_pool_provider
        )
        runner._setup()
        executor = runner.executor
        assert isinstance(executor, self.custom_pool_provider)
        assert executor.submit(sum, [1, 2]).result() == 3
        runner._shutdown_executor()

    def test_create_executor_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created_executors: List[Executor] = []

        def create_executor() -> Executor:
            executor = ThreadPoolExecutor(max_workers=1)
            created_executors.append(executor)
            return executor

        setting = copy.deepcopy(self.default_setting)
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        monkeypatch.setattr(parallel_runner, "_create_executor", create_executor)
        sequential_runner._setup()
        parallel_runner._setup()
        assert created_executors == [parallel_runner.executor]
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="FCNAgent",
        )
        assert parallel_runner.executor is None
        assert parallel_runner._get_executor() is created_executors[-1]
        assert len(created_executors) == 2
        parallel_runner._shutdown_executor()

    def _run_with_worker_initializer(
        self, setting: Dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = uuid.uuid4().hex
        setting = copy.deepcopy(setting)
        setting["FCNAgents"]["class"] = "WorkerInitializationCheckingAgent"
        setting["FCNAgents"]["workerToken"] = token
        setting["simulation"]["numParallel"] = 2
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = 6
        _, runner = self._make_runners(setting=setting)
        monkeypatch.setattr(
            runner, "_get_worker_initializer", lambda: initialize_worker
        )
        monkeypatch.setattr(runner, "_get_worker_initargs", lambda: (token,))
        runner._setup()
        runner._run()
        logger = runner.logger
        assert isinstance(logger, DummyLogger2)
        assert logger.n_order_log > 0
        assert runner.executor is None
        # the initializer is not called on the main thread
        assert get_worker_token() is None

    def test_worker_initializer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._run_with_worker_initializer(
            setting=self.default_setting, monkeypatch=monkeypatch
        )

    def test_worker_initializer_default(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "WorkerInitializationCheckingAgent"
        setting["FCNAgents"]["workerToken"] = uuid.uuid4().hex
        setting["simulation"]["sessions"][0]["iterationSteps"] = 1
        _, runner = self._make_runners(setting=setting)
        assert runner._get_worker_initializer() is None
        assert isinstance(runner._get_worker_initargs(), tuple)
        assert not runner._get_worker_initargs()
        # without the initializer, WorkerInitializationCheckingAgent fails
        runner._setup()
        with pytest.raises(RuntimeError, match="worker is initialized with None"):
            runner._run()
        assert runner.executor is None

    @pytest.mark.parametrize(
        "initializer, error_class",
        [
            (fail_to_initialize_worker, RuntimeError),
            (fail_to_initialize_worker_with_system_exit, SystemExit),
            (fail_to_initialize_worker_with_base_exception, WorkerInitializerAbort),
        ],
        ids=["Exception", "SystemExit", "BaseException"],
    )
    def test_worker_initializer_failure(
        self,
        initializer: Callable[[], None],
        error_class: Type[BaseException],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"][0]["iterationSteps"] = 1
        _, runner = self._make_runners(setting=setting)
        monkeypatch.setattr(runner, "_get_worker_initializer", lambda: initializer)
        runner._setup()
        # the tasks raise the error instead of breaking the executor, which can hang
        # ProcessPoolExecutor on Python 3.10 or earlier. The executors break on any
        # BaseException of the initializer, e.g., SystemExit, so check that the executor
        # still runs a task that does not depend on the initializer.
        executor = runner.executor
        assert executor is not None
        assert executor.submit(sum, [1, 2]).result() == 3
        with pytest.raises(
            RuntimeError, match="the worker initializer failed on this worker"
        ) as exc_info:
            runner._run()
        assert not isinstance(exc_info.value, BrokenExecutor)
        # the cause is the error itself (thread) or its traceback (process)
        cause = exc_info.value.__cause__
        assert cause is not None
        assert f"{error_class.__name__}: error in worker initializer" in "".join(
            traceback.format_exception_only(type(cause), cause)
        )
        assert runner.executor is None

    def test_split_agents_into_chunks(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        assert isinstance(runner, MultiThreadAgentParallelRunner)
        runner._setup()
        runner._shutdown_executor()
        agents = runner.simulator.normal_frequency_agents
        assert runner._split_agents_into_chunks(agents=agents) == [
            [agent] for agent in agents
        ]
        assert runner._split_agents_into_chunks(agents=[]) == []

    def test_split_agents_into_chunks_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"][0]["iterationSteps"] = 10
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        split_batches: List[List[Agent]] = []

        def split_agents_into_chunks(agents: List[Agent]) -> List[List[Agent]]:
            split_batches.append(agents)
            return [agents[:1], [], agents[1:]]

        monkeypatch.setattr(
            parallel_runner, "_split_agents_into_chunks", split_agents_into_chunks
        )
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()
        assert len(split_batches) >= 10
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="FCNAgent",
        )

    @pytest.mark.parametrize(
        "split_agents_into_chunks",
        [
            lambda agents: [agents[1:], agents[:1]],
            lambda agents: [agents[::2], agents[1::2]],
            lambda agents: [agents[1:]],
            lambda agents: [agents, agents[:1]],
            lambda agents: [agents[:1], [copy.copy(agent) for agent in agents[1:]]],
        ],
        ids=["rotated", "interleaved", "missing", "duplicated", "copied"],
    )
    def test_split_agents_into_chunks_invalid(
        self,
        split_agents_into_chunks: Callable[[List[Agent]], List[List[Agent]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # chunks that change the order of the agents would change the results silently
        _, runner = self._make_runners(setting=self.default_setting)
        monkeypatch.setattr(
            runner, "_split_agents_into_chunks", split_agents_into_chunks
        )
        runner._setup()
        with pytest.raises(ValueError, match="_split_agents_into_chunks"):
            runner._run()
        assert runner.executor is None

    def _learning_setting(
        self,
        num_parallel: int = 2,
        max_normal_orders: int = 3,
        synced_attributes: Optional[Any] = None,
    ) -> Dict:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "LearningAgent"
        if synced_attributes is not None:
            setting["FCNAgents"]["syncedAttributes"] = synced_attributes
        setting["simulation"]["numParallel"] = num_parallel
        setting["simulation"]["sessions"][0]["iterationSteps"] = 20
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = max_normal_orders
        return setting

    @pytest.mark.parametrize("num_parallel", [2, 3])
    @pytest.mark.parametrize("max_normal_orders", [2, 5])
    def test_synced_attributes_same_result_as_sequential(
        self, num_parallel: int, max_normal_orders: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setting = self._learning_setting(
            num_parallel=num_parallel, max_normal_orders=max_normal_orders
        )
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        received = _track_received_results(
            runner=parallel_runner, monkeypatch=monkeypatch
        )
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()

        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="LearningAgent",
        )
        agents = parallel_runner.simulator.agents
        assert _learning_states(sequential_runner.simulator.agents) == _learning_states(
            agents
        )
        n_calls: List[int] = []
        for agent in agents:
            assert isinstance(agent, LearningAgent)
            # the two attributes still refer to the same list
            assert agent.same_weights is agent.weights
            n_calls.append(agent.n_calls)
        # some agents deleted bias (every fourth call) and were asked again, and some agents
        # reset last_price to None (every fifth call)
        assert max(n_calls) >= 5
        results = [event for event in received if event[0] != "attributes"]
        assert any(kind == "no orders" for kind, _ in results)
        if self.receives_synced_attributes:
            # the attributes are received before the orders, even if there are no orders
            assert received == [
                event
                for kind, agent_id in results
                for event in [("attributes", agent_id), (kind, agent_id)]
            ]
        else:
            assert received == results

    def test_synced_attributes_not_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # this runner calls the agents themselves, so their changes are kept anyway
        setting = self._learning_setting(synced_attributes=[])
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        received = _track_received_results(
            runner=parallel_runner, monkeypatch=monkeypatch
        )
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="LearningAgent",
        )
        assert _learning_states(sequential_runner.simulator.agents) == _learning_states(
            parallel_runner.simulator.agents
        )
        assert len(received) > 0
        assert all(kind != "attributes" for kind, _ in received)

    @pytest.mark.parametrize(
        "synced_attributes, message",
        [
            (["simulator"], "must not include 'simulator'"),
            (["n_calls", "logger"], "must not include 'logger'"),
            (("prng",), "must not include 'prng'"),
            (["n_calls", 1], "must be a tuple or list of str"),
            ("n_calls", "must be a tuple or list of str"),
        ],
        ids=["simulator", "logger", "prng", "non-str", "str"],
    )
    def test_synced_attributes_invalid(
        self, synced_attributes: Any, message: str
    ) -> None:
        # this runner does not use synced_attributes, so it does not check them
        setting = self._learning_setting(synced_attributes=synced_attributes)
        setting["simulation"]["sessions"][0]["iterationSteps"] = 2
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        agent = runner.simulator.normal_frequency_agents[0]
        with pytest.raises(
            ValueError, match=re.escape(f"LearningAgent.synced_attributes {message}")
        ):
            _check_synced_attributes(agent=agent)
        runner._run()
        assert runner.executor is None

    def test_synced_attributes_changed_after_setup(self) -> None:
        setting = self._learning_setting()
        setting["simulation"]["sessions"][0]["iterationSteps"] = 2
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        for agent in runner.simulator.normal_frequency_agents:
            agent.synced_attributes = ("n_calls", "simulator")
        if self.receives_synced_attributes:
            # the names are checked again whenever the agents are asked
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "LearningAgent.synced_attributes must not include 'simulator'"
                ),
            ):
                runner._run()
        else:
            runner._run()
        assert runner.executor is None
        assert all(
            agent.simulator is runner.simulator for agent in runner.simulator.agents
        )

    def test_synced_attributes_set_on_class_at_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the names are sent from the main process, so the worker processes started by spawn,
        # which import the class without the change, send back the attributes too
        monkeypatch.setattr(
            UnsyncedLearningAgent, "synced_attributes", LearningAgent.synced_attributes
        )
        setting = self._learning_setting()
        setting["FCNAgents"]["class"] = "UnsyncedLearningAgent"
        setting["simulation"]["startMethod"] = "spawn"
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="UnsyncedLearningAgent",
        )
        assert _learning_states(sequential_runner.simulator.agents) == _learning_states(
            parallel_runner.simulator.agents
        )

    def test_submit_orders_in_worker_synced_attributes(self) -> None:
        setting = self._learning_setting()
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        runner._shutdown_executor()
        simulator = runner.simulator
        markets = simulator.markets
        simulator._update_times_on_markets(markets)
        agent, other_agent = simulator.normal_frequency_agents[:2]
        assert isinstance(agent, LearningAgent)
        assert isinstance(other_agent, LearningAgent)
        names = tuple(LearningAgent.synced_attributes)
        # the third call submits no orders, and the fourth call deletes bias
        agent.n_calls = 2
        data = _submit_orders_in_worker(
            agents=[agent, other_agent], markets=markets, synced_attributes=[names, ()]
        )
        # the results of all the agents are pickled together if an agent lists names
        assert isinstance(data, bytes)
        results = _load_results(results=data, simulator=simulator)
        assert len(results) == 2
        orders, prng_state, attributes = results[0]
        assert orders == []
        assert prng_state == agent.prng.getstate()
        assert attributes == {name: getattr(agent, name) for name in names}
        assert attributes is not None
        assert attributes["weights"] is attributes["same_weights"]
        # an agent without names sends back nothing, even if its class lists attributes
        assert len(results[1][0]) == 1
        assert results[1][2] is None
        # a listed attribute that the agent does not have is left out
        results = _load_results(
            results=_submit_orders_in_worker(
                agents=[agent], markets=markets, synced_attributes=[names]
            ),
            simulator=simulator,
        )
        assert len(results) == 1
        orders, _, attributes = results[0]
        assert len(orders) == 1
        assert not hasattr(agent, "bias")
        assert attributes is not None
        assert "bias" not in attributes
        assert attributes["n_calls"] == 4
        # the given names are used instead of those of the agent
        results = _load_results(
            results=_submit_orders_in_worker(
                agents=[agent], markets=markets, synced_attributes=[("n_calls",)]
            ),
            simulator=simulator,
        )
        assert len(results) == 1
        assert results[0][2] == {"n_calls": 5}
        # nothing is sent back without synced_attributes, which is the default, and the
        # results are not pickled
        plain_results = _submit_orders_in_worker(agents=[agent], markets=markets)
        assert isinstance(plain_results, list)
        assert len(plain_results) == 1
        assert plain_results[0][2] is None
        assert agent.n_calls == 6

    def _reference_setting(self, agent_class: str) -> Dict:
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = agent_class
        setting["simulation"]["numParallel"] = 2
        setting["simulation"]["sessions"][0]["iterationSteps"] = 20
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = 5
        return setting

    def test_submit_orders_in_worker_sends_back_agent_as_reference(self) -> None:
        setting = self._reference_setting(agent_class="HelperAgent")
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        runner._shutdown_executor()
        simulator = runner.simulator
        simulator._update_times_on_markets(simulator.markets)
        agent = simulator.normal_frequency_agents[0]
        assert isinstance(agent, HelperAgent)
        data = _submit_orders_in_worker(
            agents=[agent], markets=simulator.markets, synced_attributes=[("helper",)]
        )
        assert isinstance(data, bytes)
        # the helper refers to the agent and so to the whole simulation, but the agent is sent
        # back as a reference instead of a copy
        assert b"HelperAgent" not in data
        assert b"Simulator" not in data
        assert len(data) * 5 < len(
            pickle.dumps(agent.helper, protocol=pickle.HIGHEST_PROTOCOL)
        )
        results = _load_results(results=data, simulator=simulator)
        assert len(results) == 1
        orders, prng_state, attributes = results[0]
        assert len(orders) == 1
        assert prng_state == agent.prng.getstate()
        assert attributes is not None
        helper = attributes["helper"]
        assert helper is not agent.helper
        assert helper.agent is agent
        assert helper.prices == agent.helper.prices

    def test_submit_orders_in_worker_sends_back_orders_as_references(self) -> None:
        setting = self._reference_setting(agent_class="OrderTrackingAgent")
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        runner._shutdown_executor()
        simulator = runner.simulator
        markets = simulator.markets
        market = markets[0]
        simulator._update_times_on_markets(markets)
        agent, other_agent = simulator.normal_frequency_agents[:2]
        assert isinstance(agent, OrderTrackingAgent)
        assert isinstance(other_agent, OrderTrackingAgent)
        # two orders in the order book and a canceled one, which is in no order book
        for price in [290.0, 291.0, 292.0]:
            order = Order(
                agent_id=agent.agent_id,
                market_id=market.market_id,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=price,
                ttl=None,
            )
            market._add_order(order=order)
            agent.my_orders.append(order)
        market._cancel_order(cancel=Cancel(order=agent.my_orders[2]))
        my_orders_before = list(agent.my_orders)
        names = ("my_orders",)
        data = _submit_orders_in_worker(
            agents=[agent, other_agent],
            markets=markets,
            synced_attributes=[names, names],
        )
        assert isinstance(data, bytes)
        results = _load_results(results=data, simulator=simulator)
        assert len(results) == 2
        orders, _, attributes = results[0]
        assert attributes is not None
        my_orders = attributes["my_orders"]
        assert my_orders is not agent.my_orders
        assert len(my_orders) == 4
        # the orders in the order book are the objects on the main process
        assert my_orders[0] is my_orders_before[0]
        assert my_orders[1] is my_orders_before[1]
        # the canceled order is a copy
        assert my_orders[2] is not my_orders_before[2]
        assert my_orders[2] == my_orders_before[2]
        assert my_orders[2].is_canceled
        # the oldest order in the order book is canceled, and the new order, which is also
        # kept in the listed list, is one object
        assert len(orders) == 2
        cancel, new_order = orders
        assert isinstance(cancel, Cancel)
        assert cancel.order is my_orders_before[0]
        assert new_order is my_orders[3]
        assert new_order is not agent.my_orders[3]
        other_orders, _, other_attributes = results[1]
        assert other_attributes is not None
        assert len(other_orders) == 1
        assert other_orders[0] is other_attributes["my_orders"][0]

    def test_load_worker_results(self) -> None:
        setting = self._reference_setting(agent_class="OrderTrackingAgent")
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        runner._shutdown_executor()
        simulator = runner.simulator
        market = simulator.markets[0]
        simulator._update_times_on_markets(simulator.markets)
        order = Order(
            agent_id=0,
            market_id=market.market_id,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=290.0,
            ttl=None,
        )
        market._add_order(order=order)
        buffer = io.BytesIO()
        _WorkerResultPickler(buffer, simulator=simulator).dump(
            [order, market, simulator.sessions[0], simulator.agents[1].prng]
        )
        data = buffer.getvalue()
        loaded: List[Any] = _load_results(results=data, simulator=simulator)
        assert loaded[0] is order
        assert loaded[1] is market
        assert loaded[2] is simulator.sessions[0]
        assert loaded[3] is simulator.agents[1].prng
        # the order is no longer in the order book, as if a worker had added it
        market._cancel_order(cancel=Cancel(order=order))
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "A worker process sent back a reference to"
                f" ('order', {market.market_id}, True, {order.order_id}), an object of the"
                " simulation that the main process does not have."
            ),
        ):
            _load_results(results=data, simulator=simulator)
        # the objects are looked up only if the results refer to one of them
        buffer = io.BytesIO()
        _WorkerResultPickler(buffer, simulator=simulator).dump([order])
        objects = _SimulationObjects(simulator=simulator)
        loaded = _load_worker_results(results=buffer.getvalue(), objects=objects)
        assert loaded[0] is not order
        assert loaded[0] == order
        assert objects._objects is None

    @pytest.mark.parametrize(
        "agent_class", ["HelperAgent", "MarketReferencingAgent", "OrderTrackingAgent"]
    )
    def test_synced_references_same_result_as_sequential(
        self, agent_class: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setting = self._reference_setting(agent_class=agent_class)
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        receive_orders = parallel_runner._receive_orders_from_worker
        n_checked_orders: List[int] = []

        def receive_orders_from_worker(
            agent: Agent, orders: List[Union[Order, Cancel]], prng_state: Any
        ) -> List[Union[Order, Cancel]]:
            if isinstance(agent, OrderTrackingAgent):
                # the new order is the last order in the listed list, which is already set on
                # the agent, and the canceled orders are the orders in the order books
                placed_orders = _placed_orders(simulator=parallel_runner.simulator)
                assert orders[-1] is agent.my_orders[-1]
                for order in orders[:-1]:
                    assert isinstance(order, Cancel)
                    assert any(order.order is placed for placed in placed_orders)
                    assert any(order.order is mine for mine in agent.my_orders)
                n_checked_orders.append(len(orders))
            return receive_orders(agent=agent, orders=orders, prng_state=prng_state)

        monkeypatch.setattr(
            parallel_runner, "_receive_orders_from_worker", receive_orders_from_worker
        )
        # count the batches and the tables of the objects of the simulation built on the main
        # process
        split_agents_into_chunks = parallel_runner._split_agents_into_chunks
        n_batches: List[int] = []

        def count_batches(agents: List[Agent]) -> List[List[Agent]]:
            n_batches.append(len(agents))
            return split_agents_into_chunks(agents=agents)

        monkeypatch.setattr(parallel_runner, "_split_agents_into_chunks", count_batches)
        iter_simulation_objects = agent_parallel._iter_simulation_objects
        n_tables: List[int] = []

        def count_tables(simulator: Simulator) -> Iterator[Tuple[Any, Tuple[Any, ...]]]:
            n_tables.append(os.getpid())
            return iter_simulation_objects(simulator=simulator)

        monkeypatch.setattr(agent_parallel, "_iter_simulation_objects", count_tables)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()

        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class=agent_class,
        )
        agents = parallel_runner.simulator.normal_frequency_agents
        assert _reference_states(
            sequential_runner.simulator.normal_frequency_agents
        ) == _reference_states(agents)
        _assert_references_to_simulation(simulator=sequential_runner.simulator)
        _assert_references_to_simulation(simulator=parallel_runner.simulator)
        assert len(n_batches) >= 20
        if not self.receives_synced_attributes:
            assert len(n_tables) == 0
        elif agent_class == "OrderTrackingAgent":
            # no table is built for the first batch, whose results refer to no order in the
            # order books
            assert 0 < len(n_tables) < len(n_batches)
        else:
            # at most one table is built for each batch
            assert len(n_tables) == len(n_batches)
        if agent_class == "MarketReferencingAgent":
            assert all(
                isinstance(agent, MarketReferencingAgent)
                and agent.last_market is not None
                for agent in agents
            )
        if agent_class == "OrderTrackingAgent":
            assert len(n_checked_orders) == 5 * 20
            assert max(n_checked_orders) == 2
            # the orders that are no longer in the order books are executed, canceled, or
            # expired
            placed_orders = _placed_orders(simulator=parallel_runner.simulator)
            removed_orders = [
                order
                for agent in agents
                if isinstance(agent, OrderTrackingAgent)
                for order in agent.my_orders
                if not any(order is placed for placed in placed_orders)
            ]
            assert any(order.volume == 0 for order in removed_orders)
            assert any(order.is_canceled for order in removed_orders)
            assert any(
                order.volume > 0 and not order.is_canceled for order in removed_orders
            )

    def test_receive_synced_attributes_from_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setting = self._learning_setting()
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        runner._shutdown_executor()
        agent = runner.simulator.normal_frequency_agents[0]
        assert isinstance(agent, LearningAgent)
        agent.bias = 0.5
        cash_amount = agent.cash_amount
        weights = [2.0]
        runner._receive_synced_attributes_from_worker(
            agent=agent,
            attributes={
                "n_calls": 3,
                "prices": [1.0],
                "weights": weights,
                "same_weights": weights,
                "cash_amount": cash_amount + 1.0,
            },
        )
        assert agent.n_calls == 3
        assert agent.prices == [1.0]
        assert agent.weights is weights
        assert agent.same_weights is weights
        # listed attributes missing on the worker are deleted; unlisted ones are kept
        assert not hasattr(agent, "bias")
        assert not hasattr(agent, "last_price")
        assert agent.cash_amount == cash_amount
        # falsy values are set as they are
        runner._receive_synced_attributes_from_worker(
            agent=agent,
            attributes={"n_calls": 0, "prices": [], "last_price": None, "bias": 0.0},
        )
        assert agent.n_calls == 0
        assert agent.prices == []
        assert agent.last_price is None
        assert agent.bias == 0.0
        # deleting an attribute that the agent does not have is not an error
        runner._receive_synced_attributes_from_worker(agent=agent, attributes={})
        assert all(not hasattr(agent, name) for name in LearningAgent.synced_attributes)
        assert agent.cash_amount == cash_amount
        # an attribute that only the class has is kept, e.g., a class attribute set at runtime
        # that a worker started by spawn does not have
        monkeypatch.setattr(LearningAgent, "bias", 1.0, raising=False)
        agent.bias = 0.5
        runner._receive_synced_attributes_from_worker(agent=agent, attributes={})
        assert "bias" not in vars(agent)
        assert agent.bias == 1.0
        runner._receive_synced_attributes_from_worker(agent=agent, attributes={})
        assert agent.bias == 1.0

    def _record_batches(
        self, runner: MultiThreadAgentParallelRunner, monkeypatch: pytest.MonkeyPatch
    ) -> List[bool]:
        """Record whether all the tasks of the batch are finished when a result is received.

        The executor must be created before this method is called.
        """
        executor = runner.executor
        assert executor is not None
        batch_futures: List[Future] = []
        all_finished: List[bool] = []
        split_agents_into_chunks = runner._split_agents_into_chunks
        submit = executor.submit
        receive_synced_attributes = runner._receive_synced_attributes_from_worker
        receive_orders = runner._receive_orders_from_worker

        def split_agents_into_chunks_of_new_batch(
            agents: List[Agent],
        ) -> List[List[Agent]]:
            # called once for each batch before its tasks are submitted
            batch_futures.clear()
            return split_agents_into_chunks(agents=agents)

        def submit_and_record(
            fn: Callable[..., Any], *args: Any, **kwargs: Any
        ) -> Future:
            future = submit(fn, *args, **kwargs)
            batch_futures.append(future)
            return future

        def receive_synced_attributes_from_worker(
            agent: Agent, attributes: Dict[str, Any]
        ) -> None:
            all_finished.append(all(future.done() for future in batch_futures))
            receive_synced_attributes(agent=agent, attributes=attributes)

        def receive_orders_from_worker(
            agent: Agent, orders: List[Union[Order, Cancel]], prng_state: Any
        ) -> List[Union[Order, Cancel]]:
            all_finished.append(all(future.done() for future in batch_futures))
            return receive_orders(agent=agent, orders=orders, prng_state=prng_state)

        monkeypatch.setattr(
            runner, "_split_agents_into_chunks", split_agents_into_chunks_of_new_batch
        )
        monkeypatch.setattr(executor, "submit", submit_and_record)
        monkeypatch.setattr(
            runner,
            "_receive_synced_attributes_from_worker",
            receive_synced_attributes_from_worker,
        )
        monkeypatch.setattr(
            runner, "_receive_orders_from_worker", receive_orders_from_worker
        )
        return all_finished

    def test_results_are_received_after_all_tasks_are_finished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the process executor pickles the tasks, which contain all the agents, on a background
        # thread, so no agent must be updated while a task of the batch is not finished
        setting = self._learning_setting(num_parallel=2, max_normal_orders=5)
        setting["FCNAgents"]["class"] = "SlowLearningAgent"
        setting["simulation"]["sessions"][0]["iterationSteps"] = 2
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        sequential_runner._setup()
        parallel_runner._setup()
        all_finished = self._record_batches(
            runner=parallel_runner, monkeypatch=monkeypatch
        )
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="SlowLearningAgent",
        )
        assert _learning_states(sequential_runner.simulator.agents) == _learning_states(
            parallel_runner.simulator.agents
        )
        # 5 agents are asked in each of the 2 steps
        assert len(all_finished) == (20 if self.receives_synced_attributes else 10)
        assert all(all_finished)

    def test_no_agent_is_updated_if_a_task_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setting = self._learning_setting(num_parallel=2, max_normal_orders=5)
        setting["FCNAgents"]["class"] = "SlowLearningAgent"
        setting["FCNAgents"]["numAgents"] = 4
        setting["RaisingAgents"] = {
            "class": "RaisingAgent",
            "numAgents": 1,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
        }
        setting["simulation"]["agents"] = ["FCNAgents", "RaisingAgents"]
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        all_finished = self._record_batches(runner=runner, monkeypatch=monkeypatch)
        agents = runner.simulator.normal_frequency_agents
        assert all(isinstance(agent, SlowLearningAgent) for agent in agents[:4])
        states_before = _agent_states(agents)
        learning_states_before = _learning_states(agents[:4])
        # all the 5 agents are in the first batch, so the error is raised before any result
        # is received
        with pytest.raises(RuntimeError, match="error in submit_orders"):
            runner._run()
        assert runner.executor is None
        assert not all_finished
        if self.receives_synced_attributes:
            # the agents on the main process are not updated
            assert _agent_states(agents) == states_before
            assert _learning_states(agents[:4]) == learning_states_before


class TestMultiProcessAgentParallelRunner(TestMultiThreadAgentParallelRunner):
    runner_class: Type[SequentialRunner] = MultiProcessAgentParallelRunner
    custom_pool_provider: Type[Executor] = CustomProcessPoolExecutor
    receives_synced_attributes: bool = True
    TIME_PER_STEP_THRESHOLD: Optional[float] = None
    # because of the cost of pickling, the time per step is not guaranteed
    # to be less than the threshold.

    def test_collect_orders_from_normal_agents(self) -> None:
        pytest.skip(
            "mock.patch cannot be used with ProcessPoolExecutor because MagicMock is not picklable"
        )

    def test_collect_orders_from_normal_agents_error_1(self) -> None:
        pytest.skip(
            "mock.patch cannot be used with ProcessPoolExecutor because MagicMock is not picklable"
        )

    def test_prng_state_is_synced_from_worker(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = 3
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        assert isinstance(runner, MultiProcessAgentParallelRunner)
        runner._setup()
        runner.simulator._update_times_on_markets(runner.simulator.markets)
        states_before = _agent_states(runner.simulator.agents)
        orders = runner._collect_orders_from_normal_agents(
            session=runner.simulator.sessions[0]
        )
        assert len(orders) == 3
        states_after = _agent_states(runner.simulator.agents)
        n_advanced = sum(
            before["prng_state"] != after["prng_state"]
            for before, after in zip(states_before, states_after)
        )
        assert n_advanced == 3
        runner._shutdown_executor()

    def test_synced_attributes_not_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # without synced_attributes, the changes made on the worker processes are discarded
        setting = self._learning_setting(synced_attributes=[])
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        received = _track_received_results(
            runner=parallel_runner, monkeypatch=monkeypatch
        )
        sequential_runner._setup()
        parallel_runner._setup()
        initial_states = _learning_states(parallel_runner.simulator.agents)
        assert initial_states == _learning_states(sequential_runner.simulator.agents)
        sequential_runner._run()
        parallel_runner._run()
        assert _learning_states(parallel_runner.simulator.agents) == initial_states
        assert _learning_states(sequential_runner.simulator.agents) != initial_states
        assert len(received) > 0
        assert all(kind != "attributes" for kind, _ in received)
        # the agents decide differently from SequentialRunner, where the changes are kept
        sequential_market = sequential_runner.simulator.markets[0]
        parallel_market = parallel_runner.simulator.markets[0]
        times = range(sequential_market.get_time() + 1)
        assert sequential_market.get_market_prices(
            times
        ) != parallel_market.get_market_prices(times)

    @pytest.mark.parametrize(
        "synced_attributes, message",
        [
            (["simulator"], "must not include 'simulator'"),
            (["n_calls", "logger"], "must not include 'logger'"),
            (("prng",), "must not include 'prng'"),
            (["n_calls", 1], "must be a tuple or list of str"),
            ("n_calls", "must be a tuple or list of str"),
        ],
        ids=["simulator", "logger", "prng", "non-str", "str"],
    )
    def test_synced_attributes_invalid(
        self, synced_attributes: Any, message: str
    ) -> None:
        setting = self._learning_setting(synced_attributes=synced_attributes)
        _, runner = self._make_runners(setting=setting)
        with pytest.raises(
            ValueError, match=re.escape(f"LearningAgent.synced_attributes {message}")
        ):
            runner._setup()
        assert runner.executor is None

    def test_start_method(self) -> None:
        # the tests above must pass regardless of the start method of multiprocessing
        assert multiprocessing.get_start_method() in ["fork", "spawn", "forkserver"]

    @pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
    def test_start_method_config(
        self, start_method: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dummy, "PARENT_MARKER", "set on the main process")
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "CancelingAgent"
        setting["simulation"]["startMethod"] = start_method
        setting["simulation"]["numParallel"] = 2
        setting["simulation"]["sessions"][0]["iterationSteps"] = 10
        sequential_runner = self._make_runner(
            runner_class=SequentialRunner, setting=setting
        )
        parallel_runner = self._make_runner(
            runner_class=MultiProcessAgentParallelRunner, setting=setting
        )
        assert isinstance(parallel_runner, MultiProcessAgentParallelRunner)
        assert parallel_runner.start_method is None
        sequential_runner._setup()
        parallel_runner._setup()
        assert parallel_runner.start_method == start_method
        assert parallel_runner._get_mp_context().get_start_method() == start_method
        # module globals modified on the main process are inherited only by forked workers
        marker = parallel_runner._get_executor().submit(get_parent_marker).result()
        if start_method == "fork":
            assert marker == "set on the main process"
        else:
            assert marker is None
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="CancelingAgent",
        )

    @pytest.mark.parametrize("start_method", ["invalid", "", "SPAWN", 1, None])
    def test_start_method_invalid(self, start_method: Any) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["startMethod"] = start_method
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError, match="startMethod"):
            runner._setup()
        assert isinstance(runner, MultiProcessAgentParallelRunner)
        assert runner.executor is None

    def test_start_method_default(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        assert isinstance(runner, MultiProcessAgentParallelRunner)
        assert runner.start_method is None
        runner._setup()
        assert runner.start_method is None
        assert (
            runner._get_mp_context().get_start_method()
            == multiprocessing.get_context().get_start_method()
        )
        runner._shutdown_executor()

    def test_default_start_method_of_subclass(self) -> None:
        assert MultiProcessAgentParallelRunner.default_start_method is None
        setting = copy.deepcopy(self.default_setting)
        runner = SpawnMultiProcessAgentParallelRunner(settings=copy.deepcopy(setting))
        assert runner.start_method == "spawn"
        runner._setup()
        assert runner.start_method == "spawn"
        assert runner._get_mp_context().get_start_method() == "spawn"
        assert isinstance(runner.executor, ProcessPoolExecutor)
        runner._shutdown_executor()
        # simulation.startMethod takes precedence over the default of the class
        start_method = next(
            (
                method
                for method in multiprocessing.get_all_start_methods()
                if method != "spawn"
            ),
            "spawn",
        )
        setting["simulation"]["startMethod"] = start_method
        runner = SpawnMultiProcessAgentParallelRunner(settings=copy.deepcopy(setting))
        runner._setup()
        assert runner.start_method == start_method
        assert runner._get_mp_context().get_start_method() == start_method
        runner._shutdown_executor()

    def test_worker_initializer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # the initializer and its arguments must be passed to the workers even with spawn
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["startMethod"] = "spawn"
        self._run_with_worker_initializer(setting=setting, monkeypatch=monkeypatch)

    def test_split_agents_into_chunks(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        assert isinstance(runner, MultiProcessAgentParallelRunner)
        runner._setup()
        runner._shutdown_executor()
        agents = runner.simulator.normal_frequency_agents
        assert len(agents) == 10
        for num_parallel in [1, 2, 3, 4, 10, 12]:
            runner.num_parallel = num_parallel
            for n_agents in [0, 1, 2, 3, 5, 10]:
                chunks = runner._split_agents_into_chunks(agents=agents[:n_agents])
                assert len(chunks) == min(num_parallel, n_agents)
                assert [agent for chunk in chunks for agent in chunk] == agents[
                    :n_agents
                ]
                chunk_sizes = [len(chunk) for chunk in chunks]
                if n_agents > 0:
                    assert min(chunk_sizes) >= 1
                    assert max(chunk_sizes) - min(chunk_sizes) <= 1
        runner.num_parallel = 3
        assert [
            len(chunk) for chunk in runner._split_agents_into_chunks(agents=agents)
        ] == [4, 3, 3]
