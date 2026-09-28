import copy
import multiprocessing
import os
import random
import time
import uuid
from concurrent.futures import Executor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import pytest

from pams.agents import Agent
from pams.order import Cancel
from pams.order import Order
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners.sequential import SequentialRunner
from tests.pams.runners.test_sequential import TestSequentialRunner

from . import dummy
from .dummy import WAIT_TIME
from .dummy import CancelingAgent
from .dummy import DummyLogger2
from .dummy import FCNDelayAgent
from .dummy import IdleEvenIDFCNAgent
from .dummy import RaisingAgent
from .dummy import RandomlyIdleFCNAgent
from .dummy import WorkerInitializationCheckingAgent
from .dummy import fail_to_initialize_worker
from .dummy import get_parent_marker
from .dummy import get_worker_token
from .dummy import initialize_worker


class SpawnMultiProcessAgentParallelRunner(MultiProcessAgentParallelRunner):
    default_start_method = "spawn"


class CustomThreadPoolExecutor(ThreadPoolExecutor):
    pass


class CustomProcessPoolExecutor(ProcessPoolExecutor):
    pass


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
    ]
    custom_pool_provider: Type[Executor] = CustomThreadPoolExecutor

    def _make_runners(
        self, setting: Dict, seed: int = 42
    ) -> Tuple[SequentialRunner, MultiThreadAgentParallelRunner]:
        sequential_runner = SequentialRunner(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        parallel_runner = self.runner_class(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        assert isinstance(parallel_runner, MultiThreadAgentParallelRunner)
        for cls in self.user_classes:
            sequential_runner.class_register(cls=cls)
            parallel_runner.class_register(cls=cls)
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
        assert type(executor) is self.custom_pool_provider
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
        assert runner._get_worker_initargs() == ()
        # without the initializer, WorkerInitializationCheckingAgent fails
        runner._setup()
        with pytest.raises(RuntimeError, match="worker is initialized with None"):
            runner._run()
        assert runner.executor is None

    def test_worker_initializer_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"][0]["iterationSteps"] = 1
        _, runner = self._make_runners(setting=setting)
        monkeypatch.setattr(
            runner, "_get_worker_initializer", lambda: fail_to_initialize_worker
        )
        runner._setup()
        # the tasks raise the error instead of breaking the executor, which can hang
        # ProcessPoolExecutor on Python 3.10 or earlier
        with pytest.raises(
            RuntimeError, match="the worker initializer failed"
        ) as exc_info:
            runner._run()
        # the cause is the error itself (thread) or its traceback (process)
        assert "error in worker initializer" in str(exc_info.value.__cause__)
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


class TestMultiProcessAgentParallelRunner(TestMultiThreadAgentParallelRunner):
    runner_class: Type[SequentialRunner] = MultiProcessAgentParallelRunner
    custom_pool_provider: Type[Executor] = CustomProcessPoolExecutor
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
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
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
