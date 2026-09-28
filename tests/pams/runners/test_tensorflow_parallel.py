import copy
import importlib.util
import multiprocessing
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from unittest import mock

import pytest

from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import SequentialRunner
from pams.runners import TensorFlowAgentParallelRunner
from pams.runners.agent_parallel import _initialize_worker
from pams.runners.tensorflow_parallel import _initialize_tensorflow_worker

from .dummy import DummyLogger2
from .dummy import initialize_worker
from .tensorflow_dummy import TensorFlowAgent
from .tensorflow_dummy import TensorFlowModelHoldingAgent
from .tensorflow_dummy import TensorFlowReductionAgent
from .tensorflow_dummy import get_n_cached_price_models
from .tensorflow_dummy import get_tensorflow_config
from .tensorflow_dummy import load_price_model
from .tensorflow_dummy import run_sequential_runner
from .test_agent_parallel import _assert_same_results

# the tests using TensorFlow run only if TensorFlow is installed, e.g., by
# .github/workflows/ci-python-tensorflow.yml; the others replace it with a mock.
requires_tensorflow = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None, reason="TensorFlow is not installed"
)


@pytest.fixture
def fake_tensorflow(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Replace TensorFlow with a mock that has two GPUs."""
    tensorflow = mock.MagicMock()
    tensorflow.config.list_physical_devices.return_value = ["GPU:0", "GPU:1"]
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    return tensorflow


class ModelLoadingTensorFlowAgentParallelRunner(TensorFlowAgentParallelRunner):
    """TensorFlowAgentParallelRunner loading the model of TensorFlowAgent once per worker."""

    def _get_worker_initializer(self) -> Optional[Callable[..., Any]]:
        return load_price_model

    def _get_worker_initargs(self) -> Tuple[Any, ...]:
        return (self.settings["TensorFlowAgents"]["modelSeed"],)


class TestTensorFlowAgentParallelRunner:
    default_setting: Dict = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["TensorFlowAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 10,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                    "maxNormalOrders": 4,
                }
            ],
            "numParallel": 3,
        },
        "Market": {
            "class": "Market",
            "tickSize": 0.00001,
            "marketPrice": 300.0,
            "fundamentalVolatility": 0.001,
        },
        "TensorFlowAgents": {
            "class": "TensorFlowAgent",
            "numAgents": 10,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
            "modelSeed": 42,
            "noiseScale": 0.002,
        },
    }

    def _make_runner(
        self,
        setting: Dict,
        seed: int = 42,
        runner_class: Type[
            TensorFlowAgentParallelRunner
        ] = TensorFlowAgentParallelRunner,
    ) -> TensorFlowAgentParallelRunner:
        runner = runner_class(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        runner.class_register(cls=TensorFlowAgent)
        runner.class_register(cls=TensorFlowModelHoldingAgent)
        runner.class_register(cls=TensorFlowReductionAgent)
        return runner

    def test_import_does_not_import_tensorflow(self) -> None:
        code = (
            "import sys\n"
            "import pams\n"
            "from pams.runners import TensorFlowAgentParallelRunner\n"
            "assert 'tensorflow' not in sys.modules, 'TensorFlow is imported'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_tensorflow_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # None in sys.modules makes the import fail as if TensorFlow were not installed
        monkeypatch.setitem(sys.modules, "tensorflow", None)
        with pytest.raises(
            ImportError, match="requires TensorFlow.*pip install tensorflow"
        ) as exc_info:
            TensorFlowAgentParallelRunner(settings=copy.deepcopy(self.default_setting))
        assert isinstance(exc_info.value.__cause__, ImportError)
        # the worker initializer fails in the same way on the worker processes
        with pytest.raises(ImportError, match="requires TensorFlow"):
            _initialize_tensorflow_worker(1, 1, True, None, ())

    def test_start_method_default(self, fake_tensorflow: mock.MagicMock) -> None:
        assert TensorFlowAgentParallelRunner.default_start_method == "spawn"
        runner = self._make_runner(setting=self.default_setting)
        assert isinstance(runner, MultiProcessAgentParallelRunner)
        assert runner.start_method == "spawn"
        runner._setup()
        assert runner.start_method == "spawn"
        assert runner._get_mp_context().get_start_method() == "spawn"
        runner._shutdown_executor()

    @pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
    def test_start_method_config(
        self, start_method: str, fake_tensorflow: mock.MagicMock
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["startMethod"] = start_method
        runner = self._make_runner(setting=setting)
        runner._setup()
        assert runner.start_method == start_method
        assert runner._get_mp_context().get_start_method() == start_method
        runner._shutdown_executor()

    @pytest.mark.parametrize(
        "cpu_count,num_parallel,max_normal_orders,expected_threads",
        [
            (12, 1, [4], 12),
            (12, 2, [4], 6),
            (12, 3, [4], 4),
            # numParallel is 11 (the number of CPUs minus 1) by default
            (12, None, [4], 3),
            (12, 5, [7], 2),
            # at most maxNormalOrders worker processes run at the same time
            (12, 3, [1], 12),
            (12, 5, [2], 6),
            (12, 5, [1, 2], 6),
            (2, 3, [4], 1),
            (None, 3, [4], 1),
        ],
    )
    def test_config_default(
        self,
        cpu_count: Optional[int],
        num_parallel: Optional[int],
        max_normal_orders: List[int],
        expected_threads: int,
        fake_tensorflow: mock.MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
        setting = copy.deepcopy(self.default_setting)
        if num_parallel is None:
            del setting["simulation"]["numParallel"]
        else:
            setting["simulation"]["numParallel"] = num_parallel
        session_setting = setting["simulation"]["sessions"][0]
        setting["simulation"]["sessions"] = [
            dict(session_setting, sessionName=i, maxNormalOrders=value)
            for i, value in enumerate(max_normal_orders)
        ]
        runner = self._make_runner(setting=setting)
        assert runner.intra_op_parallelism_threads is None
        assert runner.inter_op_parallelism_threads == 1
        assert runner.gpu_memory_growth is True
        runner._setup()
        runner._shutdown_executor()
        assert runner.intra_op_parallelism_threads is None
        # the CPUs are divided among the worker processes running at the same time
        assert runner._get_intra_op_parallelism_threads() == expected_threads

    def test_config(self, fake_tensorflow: mock.MagicMock) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["tensorflowIntraOpThreads"] = 3
        setting["simulation"]["tensorflowInterOpThreads"] = 2
        setting["simulation"]["tensorflowGpuMemoryGrowth"] = False
        runner = self._make_runner(setting=setting)
        runner._setup()
        runner._shutdown_executor()
        assert runner.intra_op_parallelism_threads == 3
        assert runner.inter_op_parallelism_threads == 2
        assert runner.gpu_memory_growth is False
        assert runner._get_intra_op_parallelism_threads() == 3

    @pytest.mark.parametrize("initializer", [None, initialize_worker])
    def test_create_executor(
        self,
        initializer: Optional[Callable[..., Any]],
        fake_tensorflow: mock.MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["tensorflowIntraOpThreads"] = 3
        setting["simulation"]["tensorflowInterOpThreads"] = 2
        setting["simulation"]["tensorflowGpuMemoryGrowth"] = False
        runner = self._make_runner(setting=setting)
        assert runner._get_worker_initializer() is None
        assert runner._get_worker_initargs() == ()
        initargs: Tuple[Any, ...] = () if initializer is None else ("token",)
        monkeypatch.setattr(runner, "_get_worker_initializer", lambda: initializer)
        monkeypatch.setattr(runner, "_get_worker_initargs", lambda: initargs)
        pool_provider = mock.MagicMock()
        monkeypatch.setattr(runner, "_parallel_pool_provider", pool_provider)
        runner._setup()
        assert runner.executor is pool_provider.return_value
        runner._shutdown_executor()
        pool_provider.assert_called_once()
        kwargs = pool_provider.call_args.kwargs
        assert kwargs["max_workers"] == 3
        assert kwargs["mp_context"].get_start_method() == "spawn"
        # TensorFlow is configured before the initializer of the subclasses is called
        assert kwargs["initializer"] is _initialize_worker
        assert kwargs["initargs"] == (
            _initialize_tensorflow_worker,
            (3, 2, False, initializer, initargs),
        )

    @pytest.mark.parametrize(
        "key,value",
        [
            (key, value)
            for key in ["tensorflowIntraOpThreads", "tensorflowInterOpThreads"]
            for value in [0, -1, 1.5, "2", True, None]
        ]
        + [("tensorflowGpuMemoryGrowth", value) for value in [1, 0, "true", None]],
    )
    def test_config_invalid(
        self, key: str, value: Any, fake_tensorflow: mock.MagicMock
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"][key] = value
        runner = self._make_runner(setting=setting)
        with pytest.raises(ValueError, match=f"simulation.{key} must be"):
            runner._setup()
        assert runner.executor is None

    @pytest.mark.parametrize("gpu_memory_growth", [True, False])
    def test_initialize_tensorflow_worker(
        self, gpu_memory_growth: bool, fake_tensorflow: mock.MagicMock
    ) -> None:
        _initialize_tensorflow_worker(3, 2, gpu_memory_growth, None, ())
        threading = fake_tensorflow.config.threading
        threading.set_intra_op_parallelism_threads.assert_called_once_with(3)
        threading.set_inter_op_parallelism_threads.assert_called_once_with(2)
        set_memory_growth = fake_tensorflow.config.experimental.set_memory_growth
        if gpu_memory_growth:
            fake_tensorflow.config.list_physical_devices.assert_called_once_with("GPU")
            assert set_memory_growth.call_args_list == [
                mock.call("GPU:0", True),
                mock.call("GPU:1", True),
            ]
        else:
            set_memory_growth.assert_not_called()

    def test_initialize_tensorflow_worker_with_initializer(
        self, fake_tensorflow: mock.MagicMock
    ) -> None:
        # a child of the mock records its calls in the order of those of TensorFlow
        initializer = fake_tensorflow.initializer
        _initialize_tensorflow_worker(3, 2, True, initializer, ("token", 1))
        initializer.assert_called_once_with("token", 1)
        assert fake_tensorflow.mock_calls[-1] == mock.call.initializer("token", 1)

    def test_initialize_tensorflow_worker_after_initialization(
        self, fake_tensorflow: mock.MagicMock
    ) -> None:
        error = RuntimeError(
            "Intra op parallelism cannot be modified after initialization."
        )
        threading = fake_tensorflow.config.threading
        threading.set_intra_op_parallelism_threads.side_effect = error
        initializer = mock.MagicMock()
        with pytest.raises(RuntimeError, match="already initialized") as exc_info:
            _initialize_tensorflow_worker(1, 1, True, initializer, ())
        assert exc_info.value.__cause__ is error
        initializer.assert_not_called()

    @requires_tensorflow
    @pytest.mark.parametrize(
        "agent_class,runner_class",
        [
            ("TensorFlowAgent", TensorFlowAgentParallelRunner),
            ("TensorFlowModelHoldingAgent", TensorFlowAgentParallelRunner),
            ("TensorFlowAgent", ModelLoadingTensorFlowAgentParallelRunner),
        ],
        ids=["cached-model", "held-model", "model-loaded-by-initializer"],
    )
    def test_same_result_as_sequential(
        self, agent_class: str, runner_class: Type[TensorFlowAgentParallelRunner]
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["TensorFlowAgents"]["class"] = agent_class
        sequential_runner = SequentialRunner(
            settings=copy.deepcopy(setting),
            prng=random.Random(42),
            logger=DummyLogger2(),
        )
        sequential_runner.class_register(cls=TensorFlowAgent)
        sequential_runner.class_register(cls=TensorFlowModelHoldingAgent)
        parallel_runner = self._make_runner(setting=setting, runner_class=runner_class)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class=agent_class,
        )
        # the orders of the agents are executed, so the features given to the model change
        assert isinstance(parallel_runner.logger, DummyLogger2)
        assert parallel_runner.logger.n_execution_log > 0
        assert parallel_runner.executor is None

    @requires_tensorflow
    def test_same_result_as_sequential_with_same_threads(self) -> None:
        # TensorFlowReductionAgent reduces about a million samples on several threads, so the
        # results depend on the number of intra-op threads. SequentialRunner runs on a new process
        # whose TensorFlow is configured in the same way as the worker processes.
        n_threads = 2
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["tensorflowIntraOpThreads"] = n_threads
        # with a tiny tick size, the rounding errors of TensorFlow change the prices
        setting["Market"]["tickSize"] = 1e-9
        setting["TensorFlowAgents"]["class"] = "TensorFlowReductionAgent"
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_tensorflow_worker,
            initargs=(n_threads, 1, True, None, ()),
        ) as executor:
            sequential_runner = executor.submit(
                run_sequential_runner, copy.deepcopy(setting), 42
            ).result()
        parallel_runner = self._make_runner(setting=setting)
        parallel_runner._setup()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class="TensorFlowReductionAgent",
        )
        assert isinstance(parallel_runner.logger, DummyLogger2)
        assert parallel_runner.logger.n_execution_log > 0

    @requires_tensorflow
    @pytest.mark.parametrize(
        "threads,expected_threads",
        [(None, (os.cpu_count() or 1, 1)), ((3, 2), (3, 2))],
        ids=["default", "config"],
    )
    def test_worker_is_configured(
        self, threads: Any, expected_threads: Tuple[int, int]
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["numParallel"] = 1
        if threads is not None:
            setting["simulation"]["tensorflowIntraOpThreads"] = threads[0]
            setting["simulation"]["tensorflowInterOpThreads"] = threads[1]
        runner = self._make_runner(setting=setting)
        runner._setup()
        try:
            intra_op_threads, inter_op_threads, gpu_memory_growths = (
                runner._get_executor().submit(get_tensorflow_config).result()
            )
        finally:
            runner._shutdown_executor()
        assert (intra_op_threads, inter_op_threads) == expected_threads
        expected_growths: List[bool] = [True] * len(gpu_memory_growths)
        assert gpu_memory_growths == expected_growths

    @requires_tensorflow
    def test_worker_initializer(self) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["numParallel"] = 1
        setting["simulation"]["tensorflowIntraOpThreads"] = 3
        setting["simulation"]["tensorflowInterOpThreads"] = 2
        runner = self._make_runner(
            setting=setting, runner_class=ModelLoadingTensorFlowAgentParallelRunner
        )
        runner._setup()
        try:
            executor = runner._get_executor()
            n_cached_models = executor.submit(get_n_cached_price_models).result()
            intra_op_threads, inter_op_threads, _ = executor.submit(
                get_tensorflow_config
            ).result()
        finally:
            runner._shutdown_executor()
        # the initializer loaded the model before the first task
        assert n_cached_models == 1
        # the initializer ran TensorFlow after TensorFlow had been configured;
        # otherwise, the configuration would have failed
        assert (intra_op_threads, inter_op_threads) == (3, 2)
