import copy
import multiprocessing
import os
import random
import time
from typing import Any
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

from .dummy import CancelingAgent
from .dummy import DummyLogger2
from .dummy import FCNDelayAgent
from .dummy import IdleEvenIDFCNAgent
from .dummy import RaisingAgent
from .dummy import RandomlyIdleFCNAgent
from .dummy import wait_time


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
    ]

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
        assert orders == []
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
        # 5 steps x 3 agents x wait_time in sequential
        start_time = time.time()
        sequential_runner.main()
        end_time = time.time()
        elps_time_sequential = end_time - start_time
        # 5 steps x 1 batch x wait_time in parallel
        start_time = time.time()
        parallel_runner.main()
        end_time = time.time()
        elps_time_parallel = end_time - start_time
        overhead_time = max(elps_time_sequential - wait_time * 15, 0.0)
        assert elps_time_sequential > wait_time * 15
        assert elps_time_parallel > wait_time * 5
        assert elps_time_parallel < wait_time * 5 + overhead_time + 1
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


class TestMultiProcessAgentParallelRunner(TestMultiThreadAgentParallelRunner):
    runner_class: Type[SequentialRunner] = MultiProcessAgentParallelRunner
    TIME_PER_STEP_THRESHOLD: Optional[float] = None
    # because of the cost of pickling, the time per step is not guaranteed to be less than the threshold.

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
            [
                before["prng_state"] != after["prng_state"]
                for before, after in zip(states_before, states_after)
            ]
        )
        assert n_advanced == 3
        runner._shutdown_executor()

    def test_start_method(self) -> None:
        # the tests above must pass regardless of the start method of multiprocessing
        assert multiprocessing.get_start_method() in ["fork", "spawn", "forkserver"]
