import copy
import importlib.util
import multiprocessing
import os
import random
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from unittest import mock

import pytest

from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import SequentialRunner
from pams.runners import TensorFlowAgentParallelRunner
from pams.runners.tensorflow_parallel import _initialize_tensorflow_worker

from .dummy import DummyLogger2
from .tensorflow_dummy import TensorFlowAgent
from .tensorflow_dummy import TensorFlowModelHoldingAgent
from .tensorflow_dummy import get_tensorflow_config
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
            "noiseScale": 0.001,
        },
    }

    def _make_runner(
        self, setting: Dict, seed: int = 42
    ) -> TensorFlowAgentParallelRunner:
        runner = TensorFlowAgentParallelRunner(
            settings=copy.deepcopy(setting),
            prng=random.Random(seed),
            logger=DummyLogger2(),
        )
        runner.class_register(cls=TensorFlowAgent)
        runner.class_register(cls=TensorFlowModelHoldingAgent)
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
            _initialize_tensorflow_worker(1, 1, True)

    def test_start_method_default(self, fake_tensorflow: mock.MagicMock) -> None:
        assert TensorFlowAgentParallelRunner.default_start_method == "spawn"
        runner = self._make_runner(setting=self.default_setting)
        assert isinstance(runner, MultiProcessAgentParallelRunner)
        assert runner.start_method == "spawn"
        runner._setup()
        assert runner.start_method == "spawn"
        assert runner._get_mp_context().get_start_method() == "spawn"
        assert runner._get_worker_initializer() is _initialize_tensorflow_worker
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

    @pytest.mark.parametrize("num_parallel", [1, 2, 3, None])
    def test_config_default(
        self, num_parallel: Any, fake_tensorflow: mock.MagicMock
    ) -> None:
        setting = copy.deepcopy(self.default_setting)
        if num_parallel is None:
            del setting["simulation"]["numParallel"]
        else:
            setting["simulation"]["numParallel"] = num_parallel
        runner = self._make_runner(setting=setting)
        assert runner.intra_op_parallelism_threads is None
        assert runner.inter_op_parallelism_threads == 1
        assert runner.gpu_memory_growth is True
        runner._setup()
        runner._shutdown_executor()
        assert runner.intra_op_parallelism_threads is None
        # the CPUs are divided among the worker processes
        assert runner._get_worker_initargs() == (
            max((os.cpu_count() or 1) // runner.num_parallel, 1),
            1,
            True,
        )

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
        assert runner._get_worker_initargs() == (3, 2, False)

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
        _initialize_tensorflow_worker(3, 2, gpu_memory_growth)
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

    def test_initialize_tensorflow_worker_after_initialization(
        self, fake_tensorflow: mock.MagicMock
    ) -> None:
        error = RuntimeError(
            "Intra op parallelism cannot be modified after initialization."
        )
        threading = fake_tensorflow.config.threading
        threading.set_intra_op_parallelism_threads.side_effect = error
        with pytest.raises(RuntimeError, match="already initialized") as exc_info:
            _initialize_tensorflow_worker(1, 1, True)
        assert exc_info.value.__cause__ is error

    @requires_tensorflow
    @pytest.mark.parametrize(
        "agent_class", ["TensorFlowAgent", "TensorFlowModelHoldingAgent"]
    )
    def test_same_result_as_sequential(self, agent_class: str) -> None:
        setting = copy.deepcopy(self.default_setting)
        setting["TensorFlowAgents"]["class"] = agent_class
        sequential_runner = SequentialRunner(
            settings=copy.deepcopy(setting),
            prng=random.Random(42),
            logger=DummyLogger2(),
        )
        sequential_runner.class_register(cls=TensorFlowAgent)
        sequential_runner.class_register(cls=TensorFlowModelHoldingAgent)
        parallel_runner = self._make_runner(setting=setting)
        sequential_runner._setup()
        parallel_runner._setup()
        sequential_runner._run()
        parallel_runner._run()
        _assert_same_results(
            sequential_runner=sequential_runner,
            parallel_runner=parallel_runner,
            agent_class=agent_class,
        )
        assert parallel_runner.executor is None

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
