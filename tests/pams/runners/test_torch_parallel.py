import copy
import multiprocessing
import os
import pickle  # nosec B403  # only measures the size of pickled data
import random
import subprocess  # nosec B404  # runs only the Python of the tests
import sys
from concurrent.futures import Executor
from multiprocessing.reduction import ForkingPickler
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import pytest

import pams
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import SequentialRunner
from pams.runners import TorchAgentParallelRunner
from pams.runners.torch_parallel import _initialize_torch_worker

from .dummy import DummyLogger2
from .test_agent_parallel import _assert_same_results
from .torch_dummy import FakeTorch
from .torch_dummy import TorchPricingAgent
from .torch_dummy import get_torch_settings
from .torch_dummy import inspect_tensor

SETTING: Dict = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["TorchAgents", "FCNAgents"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": 20,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "maxNormalOrders": 4,
            }
        ],
        "numParallel": 2,
    },
    "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
    "TorchAgents": {
        "class": "TorchPricingAgent",
        "numAgents": 8,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
    },
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents": 4,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
}


class SharingStrategyKeepingTorchAgentParallelRunner(TorchAgentParallelRunner):
    torch_sharing_strategy: Optional[str] = None


@pytest.fixture(name="torch")
def fixture_torch() -> Iterator[Any]:
    """Import PyTorch, or skip the test if it is not installed.

    The sharing strategy of PyTorch, which the runners set on the main process, is restored after
    the test.
    """
    torch = pytest.importorskip("torch")
    sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    yield torch
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


@pytest.fixture(name="fake_torch")
def fixture_fake_torch(monkeypatch: pytest.MonkeyPatch) -> FakeTorch:
    """Replace PyTorch with a fake module using 12 threads on the main process.

    The worker processes would import the real PyTorch, so the tests using the fake module must not
    start them.
    """
    fake_torch = FakeTorch(num_threads=12)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "torch.multiprocessing", fake_torch.multiprocessing
    )
    return fake_torch


@pytest.fixture(autouse=True)
def shut_down_executors(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Shut down the executors created by the runners after each test.

    The worker processes of the executors left running, e.g., by the tests that only set up the
    runner, would otherwise keep PyTorch loaded until they are garbage-collected.
    """
    executors: List[Executor] = []
    create_executor: Callable[
        [MultiProcessAgentParallelRunner], Executor
    ] = TorchAgentParallelRunner._create_executor

    def create_and_track_executor(runner: MultiProcessAgentParallelRunner) -> Executor:
        executor = create_executor(runner)
        executors.append(executor)
        return executor

    monkeypatch.setattr(
        TorchAgentParallelRunner, "_create_executor", create_and_track_executor
    )
    yield
    for executor in executors:
        executor.shutdown(wait=True)


def test_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes the import fail whether PyTorch is installed or not
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", None)
    with pytest.raises(ImportError, match="requires PyTorch") as exc_info:
        TorchAgentParallelRunner(settings=copy.deepcopy(SETTING))
    assert "pip install torch" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)
    # the worker processes fail in the same way
    with pytest.raises(ImportError, match="requires PyTorch"):
        _initialize_torch_worker(num_threads=1, sharing_strategy=None)


def test_pams_does_not_import_torch() -> None:
    # a new process is used because PyTorch may already be imported by the other tests
    code = "import sys, pams, pams.runners; print('torch' in sys.modules)"
    result = subprocess.run(  # nosec B603  # a fixed command without user input
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(pams.__file__))),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


class TestTorchAgentParallelRunnerWithFakeTorch:
    """Tests of the runner on the main process, which do not need PyTorch."""

    def _make_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runner_class: Type[TorchAgentParallelRunner] = TorchAgentParallelRunner,
        max_normal_orders: int = 4,
        **simulation_settings: Any,
    ) -> TorchAgentParallelRunner:
        # the fake module cannot run the PyTorch agents, and the worker processes would import
        # the real PyTorch, so only FCN agents are used and no workers are started
        setting = copy.deepcopy(SETTING)
        setting["simulation"]["agents"] = ["FCNAgents"]
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = max_normal_orders
        setting["simulation"].update(simulation_settings)
        runner = runner_class(settings=setting)
        monkeypatch.setattr(runner, "_create_executor", lambda: None)
        return runner

    def test_default_start_method(
        self, fake_torch: FakeTorch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = self._make_runner(monkeypatch)
        assert runner.start_method == "spawn"
        runner._setup()
        assert runner._get_mp_context().get_start_method() == "spawn"
        assert runner._get_worker_initializer() is _initialize_torch_worker
        assert runner._get_worker_initargs() == (6, "file_system")
        # the main process keeps its number of threads
        assert fake_torch.num_threads == 12

    @pytest.mark.parametrize(
        "num_parallel, max_normal_orders, num_threads",
        [(1, 4, 12), (2, 4, 6), (1000, 4, 3), (8, 2, 6), (3, 1, 12), (24, 30, 1)],
    )
    @pytest.mark.usefixtures("fake_torch")
    def test_torch_num_threads_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        num_parallel: int,
        max_normal_orders: int,
        num_threads: int,
    ) -> None:
        # 12 threads are divided among the workers that can run tasks at the same time, i.e., at
        # most max_normal_orders workers, and each worker gets at least one thread
        runner = self._make_runner(
            monkeypatch, max_normal_orders=max_normal_orders, numParallel=num_parallel
        )
        # before the setup, there are no sessions and all the workers are assumed to be busy
        runner.num_parallel = num_parallel
        assert runner._get_worker_initargs()[0] == max(12 // num_parallel, 1)
        runner._setup()
        assert runner.torch_num_threads is None
        assert runner._get_worker_initargs() == (num_threads, "file_system")

    def test_torch_num_threads_config(
        self, fake_torch: FakeTorch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = self._make_runner(monkeypatch, torchNumThreads=5)
        runner._setup()
        assert runner.torch_num_threads == 5
        assert runner._get_worker_initargs() == (5, "file_system")
        assert fake_torch.num_threads == 12

    @pytest.mark.parametrize("torch_num_threads", [0, -1, 1.5, "2", True, None])
    @pytest.mark.usefixtures("fake_torch")
    def test_torch_num_threads_invalid(
        self, monkeypatch: pytest.MonkeyPatch, torch_num_threads: Any
    ) -> None:
        runner = self._make_runner(monkeypatch, torchNumThreads=torch_num_threads)
        with pytest.raises(ValueError, match="torchNumThreads"):
            runner._setup()

    def test_sharing_strategy(
        self, fake_torch: FakeTorch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = self._make_runner(monkeypatch)
        assert fake_torch.multiprocessing.get_sharing_strategy() == "file_descriptor"
        runner._setup()
        assert fake_torch.multiprocessing.get_sharing_strategy() == "file_system"
        assert runner._get_worker_initargs()[1] == "file_system"

    def test_sharing_strategy_none(
        self, fake_torch: FakeTorch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = self._make_runner(
            monkeypatch, runner_class=SharingStrategyKeepingTorchAgentParallelRunner
        )
        runner._setup()
        assert fake_torch.multiprocessing.get_sharing_strategy() == "file_descriptor"
        assert runner._get_worker_initargs()[1] is None

    def test_initialize_worker(self, fake_torch: FakeTorch) -> None:
        # the initializer runs on the worker processes, so it is called here directly
        _initialize_torch_worker(num_threads=3, sharing_strategy="file_system")
        assert fake_torch.num_threads == 3
        assert fake_torch.multiprocessing.get_sharing_strategy() == "file_system"
        # None keeps the sharing strategy
        fake_torch.multiprocessing.set_sharing_strategy("file_descriptor")
        _initialize_torch_worker(num_threads=1, sharing_strategy=None)
        assert fake_torch.num_threads == 1
        assert fake_torch.multiprocessing.get_sharing_strategy() == "file_descriptor"


@pytest.mark.usefixtures("torch")
class TestTorchAgentParallelRunner:
    def _make_runners(
        self,
        setting: Dict,
        seed: int = 42,
        runner_class: Type[TorchAgentParallelRunner] = TorchAgentParallelRunner,
    ) -> Tuple[SequentialRunner, TorchAgentParallelRunner]:
        sequential_runner = SequentialRunner(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        parallel_runner = runner_class(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        sequential_runner.class_register(cls=TorchPricingAgent)
        parallel_runner.class_register(cls=TorchPricingAgent)
        return sequential_runner, parallel_runner

    @pytest.mark.parametrize("num_parallel", [1, 3])
    @pytest.mark.parametrize("max_normal_orders", [1, 4])
    def test_same_result_as_sequential(
        self, torch: Any, num_parallel: int, max_normal_orders: int
    ) -> None:
        setting = copy.deepcopy(SETTING)
        setting["simulation"]["numParallel"] = num_parallel
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = max_normal_orders
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="TorchPricingAgent",
        )
        assert parallel_runner.executor is None
        # the models were sent to the workers through shared memory and were not modified there
        for sequential_agent, parallel_agent in zip(
            sequential_runner.simulator.normal_frequency_agents,
            parallel_runner.simulator.normal_frequency_agents,
        ):
            if isinstance(parallel_agent, TorchPricingAgent):
                assert isinstance(sequential_agent, TorchPricingAgent)
                for sequential_parameter, parallel_parameter in zip(
                    sequential_agent.model.parameters(),
                    parallel_agent.model.parameters(),
                ):
                    assert parallel_parameter.is_shared()
                    assert torch.equal(sequential_parameter, parallel_parameter)

    def test_seed_changes_result(self, torch: Any) -> None:
        # the equality above is meaningful only if the PyTorch agents depend on the seed
        prices: List[List[float]] = []
        weights: List[Any] = []
        for seed in [1, 2]:
            runner, _ = self._make_runners(setting=SETTING, seed=seed)
            runner._setup()
            runner._run()
            market = runner.simulator.markets[0]
            prices.append(market.get_market_prices(range(market.get_time() + 1)))
            agent = next(
                agent
                for agent in runner.simulator.normal_frequency_agents
                if isinstance(agent, TorchPricingAgent)
            )
            weights.append(next(agent.model.parameters()))
        assert prices[0] != prices[1]
        assert not torch.equal(weights[0], weights[1])

    def test_default_start_method(self) -> None:
        assert TorchAgentParallelRunner.default_start_method == "spawn"
        _, runner = self._make_runners(setting=SETTING)
        assert runner.start_method == "spawn"
        runner._setup()
        assert runner.start_method == "spawn"
        assert runner._get_mp_context().get_start_method() == "spawn"

    def test_start_method_config(self) -> None:
        # simulation.startMethod takes precedence over the default of the runner
        start_method = next(
            (
                method
                for method in multiprocessing.get_all_start_methods()
                if method != "spawn"
            ),
            "spawn",
        )
        setting = copy.deepcopy(SETTING)
        setting["simulation"]["startMethod"] = start_method
        _, runner = self._make_runners(setting=setting)
        runner._setup()
        assert runner.start_method == start_method
        assert runner._get_mp_context().get_start_method() == start_method

    @pytest.mark.parametrize(
        "num_parallel, max_normal_orders, num_busy_workers",
        [(1, 4, 1), (2, 4, 2), (3, 4, 3), (1000, 4, 4), (8, 2, 2), (3, 1, 1)],
    )
    def test_torch_num_threads_default(
        self,
        torch: Any,
        monkeypatch: pytest.MonkeyPatch,
        num_parallel: int,
        max_normal_orders: int,
        num_busy_workers: int,
    ) -> None:
        # the threads are divided among the workers that can run tasks at the same time, i.e.,
        # at most max_normal_orders workers
        setting = copy.deepcopy(SETTING)
        setting["simulation"]["numParallel"] = num_parallel
        setting["simulation"]["sessions"][0]["maxNormalOrders"] = max_normal_orders
        _, runner = self._make_runners(setting=setting)
        assert runner.torch_num_threads is None
        # before the setup, there are no sessions and all the workers are assumed to be busy
        runner.num_parallel = num_parallel
        assert runner._get_worker_initargs()[0] == max(
            torch.get_num_threads() // num_parallel, 1
        )
        if num_parallel > 2:
            # the workers are started only for a few values to keep the test fast
            monkeypatch.setattr(runner, "_create_executor", lambda: None)
        runner._setup()
        assert runner.torch_num_threads is None
        expected = max(torch.get_num_threads() // num_busy_workers, 1)
        assert runner._get_worker_initargs() == (expected, "file_system")
        assert runner._get_worker_initializer() is _initialize_torch_worker
        if runner.executor is not None:
            assert runner.executor.submit(get_torch_settings).result()[0] == expected

    def test_torch_num_threads_config(self, torch: Any) -> None:
        setting = copy.deepcopy(SETTING)
        setting["simulation"]["torchNumThreads"] = 2
        sequential_runner, parallel_runner = self._make_runners(setting=setting)
        main_num_threads = torch.get_num_threads()
        sequential_runner._setup()
        parallel_runner._setup()
        assert parallel_runner.torch_num_threads == 2
        assert parallel_runner._get_worker_initargs() == (2, "file_system")
        executor = parallel_runner._get_executor()
        assert executor.submit(get_torch_settings).result()[0] == 2
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="TorchPricingAgent",
        )
        # the main process is not affected
        assert torch.get_num_threads() == main_num_threads

    @pytest.mark.parametrize("torch_num_threads", [0, -1, 1.5, "2", True, None])
    def test_torch_num_threads_invalid(self, torch_num_threads: Any) -> None:
        setting = copy.deepcopy(SETTING)
        setting["simulation"]["torchNumThreads"] = torch_num_threads
        _, runner = self._make_runners(setting=setting)
        with pytest.raises(ValueError, match="torchNumThreads"):
            runner._setup()
        assert runner.executor is None

    def test_sharing_strategy(self, torch: Any) -> None:
        # file_descriptor, the default on Linux, would send a file descriptor per tensor per task
        assert TorchAgentParallelRunner.torch_sharing_strategy == "file_system"
        torch.multiprocessing.set_sharing_strategy(
            sorted(torch.multiprocessing.get_all_sharing_strategies())[0]
        )
        _, runner = self._make_runners(setting=SETTING)
        runner._setup()
        assert torch.multiprocessing.get_sharing_strategy() == "file_system"
        executor = runner._get_executor()
        assert executor.submit(get_torch_settings).result()[1] == "file_system"

    def test_sharing_strategy_none(self, torch: Any) -> None:
        # the first one in alphabetical order is file_descriptor on Linux
        sharing_strategy = sorted(torch.multiprocessing.get_all_sharing_strategies())[0]
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        sequential_runner, parallel_runner = self._make_runners(
            setting=SETTING, runner_class=SharingStrategyKeepingTorchAgentParallelRunner
        )
        sequential_runner._setup()
        parallel_runner._setup()
        assert parallel_runner._get_worker_initargs()[1] is None
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="TorchPricingAgent",
        )
        assert torch.multiprocessing.get_sharing_strategy() == sharing_strategy

    def test_tensors_are_shared_with_workers(self, torch: Any) -> None:
        # float64 makes the sum exact whatever the number of threads of the workers is
        tensor = torch.arange(100_000, dtype=torch.float64)
        # the tensors are pickled with their data by pickle, but by their handles of shared
        # memory by ForkingPickler, which ProcessPoolExecutor uses
        assert len(pickle.dumps(tensor)) > 800_000
        assert not tensor.is_shared()
        _, runner = self._make_runners(setting=SETTING)
        runner._setup()
        executor = runner._get_executor()
        is_shared, total = executor.submit(inspect_tensor, tensor).result()
        assert is_shared
        assert total == float(sum(range(100_000)))
        # the tensor on the main process is moved to shared memory
        assert tensor.is_shared()
        assert len(ForkingPickler.dumps(tensor)) < 1000
