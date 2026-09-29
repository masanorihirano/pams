import copy
import os
import pickle  # nosec B403 # only objects created by the tests are unpickled
import random
from concurrent.futures import Executor
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import pytest

from pams.runners import JaxAgentParallelRunner
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import SequentialRunner

from .dummy import DummyLogger2
from .dummy import fail_to_initialize_worker
from .test_agent_parallel import _assert_same_results

# the tests below require JAX and Flax, which are optional dependencies of pams, so jax_dummy,
# which imports them, is imported after they are found. The tests that do not require them are in
# test_jax_parallel_import.py.
jax = pytest.importorskip("jax", reason="JAX is not installed")
pytest.importorskip("flax", reason="Flax is not installed")

from . import jax_dummy  # noqa: E402  # pylint: disable=wrong-import-position

SHARED_MODEL_SEED = 1234

DEFAULT_SETTING: Dict = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FlaxAgents"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": 10,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "events": ["FundamentalPriceShock"],
                "maxNormalOrders": 4,
            }
        ],
        "numParallel": 2,
    },
    "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
    "FlaxAgents": {
        "class": "FlaxPriceAgent",
        "numAgents": 10,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
    },
    "FundamentalPriceShock": {
        "class": "FundamentalPriceShock",
        "target": "Market",
        "triggerTime": 3,
        "priceChangeRate": -0.1,
        "shockTimeLength": 1,
        "enabled": True,
    },
}


class SharedModelJaxAgentParallelRunner(JaxAgentParallelRunner):
    """JaxAgentParallelRunner loading the shared model once per worker process."""

    def _get_worker_initializer(self) -> Optional[Callable[..., Any]]:
        return jax_dummy.load_shared_model

    def _get_worker_initargs(self) -> Tuple[Any, ...]:
        return (SHARED_MODEL_SEED,)


class XlaFlagsJaxAgentParallelRunner(JaxAgentParallelRunner):
    """JaxAgentParallelRunner making JAX use two CPU devices on the worker processes."""

    def _get_worker_environment(self) -> Dict[str, str]:
        environment = super()._get_worker_environment()
        environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        return environment


class FailingJaxAgentParallelRunner(JaxAgentParallelRunner):
    def _get_worker_initializer(self) -> Optional[Callable[..., Any]]:
        return fail_to_initialize_worker


@pytest.fixture(autouse=True)
def shut_down_executors(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Shut down the executors created by the runners after each test."""
    executors: List[Executor] = []
    create_executor = JaxAgentParallelRunner._create_executor

    def create_and_track_executor(runner: JaxAgentParallelRunner) -> Executor:
        executor = create_executor(runner)
        executors.append(executor)
        return executor

    monkeypatch.setattr(
        JaxAgentParallelRunner, "_create_executor", create_and_track_executor
    )
    yield
    for executor in executors:
        executor.shutdown(wait=True)


def _make_runners(
    setting: Dict,
    runner_class: Type[JaxAgentParallelRunner] = JaxAgentParallelRunner,
    seed: int = 42,
) -> Tuple[SequentialRunner, JaxAgentParallelRunner]:
    sequential_runner = SequentialRunner(
        settings=copy.deepcopy(setting), prng=random.Random(seed), logger=DummyLogger2()
    )
    parallel_runner = runner_class(
        settings=copy.deepcopy(setting), prng=random.Random(seed), logger=DummyLogger2()
    )
    for cls in [jax_dummy.FlaxPriceAgent, jax_dummy.SharedFlaxModelAgent]:
        sequential_runner.class_register(cls=cls)
        parallel_runner.class_register(cls=cls)
    return sequential_runner, parallel_runner


def _make_runner(setting: Dict) -> JaxAgentParallelRunner:
    _, runner = _make_runners(setting=setting)
    return runner


def test_experimental_warning() -> None:
    with pytest.warns(UserWarning, match="JaxAgentParallelRunner is experimental"):
        JaxAgentParallelRunner(settings=copy.deepcopy(DEFAULT_SETTING))


def test_default_start_method() -> None:
    assert issubclass(JaxAgentParallelRunner, MultiProcessAgentParallelRunner)
    assert JaxAgentParallelRunner.default_start_method == "spawn"
    runner = _make_runner(setting=DEFAULT_SETTING)
    assert runner.start_method == "spawn"
    runner._setup()
    assert runner.start_method == "spawn"
    assert runner._get_mp_context().get_start_method() == "spawn"
    runner._shutdown_executor()


@pytest.mark.parametrize(
    "num_parallel, max_normal_orders", [(1, 4), (2, 4), (4, 4), (2, 1)]
)
def test_same_result_as_sequential(num_parallel: int, max_normal_orders: int) -> None:
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["numParallel"] = num_parallel
    setting["simulation"]["sessions"][0]["maxNormalOrders"] = max_normal_orders
    sequential_runner, parallel_runner = _make_runners(setting=setting)
    sequential_runner._setup()
    parallel_runner._setup()
    sequential_runner._run()
    parallel_runner._run()
    _assert_same_results(
        sequential_runner=sequential_runner,
        parallel_runner=parallel_runner,
        agent_class="FlaxPriceAgent",
    )
    logger = parallel_runner.logger
    assert isinstance(logger, DummyLogger2)
    assert logger.n_execution_log > 0
    assert parallel_runner.executor is None


def test_same_result_as_sequential_with_two_markets() -> None:
    # each agent submits orders only to the market accessible to it
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["markets"] = ["Market", "OtherMarket"]
    setting["simulation"]["agents"] = ["FlaxAgents", "OtherFlaxAgents"]
    setting["OtherMarket"] = {"extends": "Market", "marketPrice": 200.0}
    setting["OtherFlaxAgents"] = {"extends": "FlaxAgents", "markets": ["OtherMarket"]}
    sequential_runner, parallel_runner = _make_runners(setting=setting)
    sequential_runner._setup()
    parallel_runner._setup()
    sequential_runner._run()
    parallel_runner._run()
    _assert_same_results(
        sequential_runner=sequential_runner,
        parallel_runner=parallel_runner,
        agent_class="FlaxPriceAgent",
    )
    sequential_market = sequential_runner.simulator.markets[1]
    parallel_market = parallel_runner.simulator.markets[1]
    assert parallel_market.name == "OtherMarket"
    times = range(sequential_market.get_time() + 1)
    assert sequential_market.get_market_prices(
        times
    ) == parallel_market.get_market_prices(times)
    assert sequential_market.get_executed_volumes(
        times
    ) == parallel_market.get_executed_volumes(times)
    assert sum(parallel_market.get_executed_volumes(times)) > 0


def test_same_result_as_sequential_with_shared_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # restore the state of the main process after the test
    monkeypatch.setattr(jax_dummy.PROCESS_STATE, "shared_params", None)
    monkeypatch.setattr(jax_dummy.PROCESS_STATE, "preallocate_at_load", None)
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["FlaxAgents"]["class"] = "SharedFlaxModelAgent"
    sequential_runner, parallel_runner = _make_runners(
        setting=setting, runner_class=SharedModelJaxAgentParallelRunner
    )
    jax_dummy.load_shared_model(seed=SHARED_MODEL_SEED)
    sequential_runner._setup()
    parallel_runner._setup()
    # the worker initializer of the subclass is called after JAX is configured
    config = parallel_runner._get_executor().submit(jax_dummy.get_jax_worker_config)
    assert config.result()["preallocate_at_load"] == "false"
    sequential_runner._run()
    parallel_runner._run()
    _assert_same_results(
        sequential_runner=sequential_runner,
        parallel_runner=parallel_runner,
        agent_class="SharedFlaxModelAgent",
    )


def test_shared_model_not_copied_to_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # restore the state of the main process after the test
    monkeypatch.setattr(jax_dummy.PROCESS_STATE, "shared_params", None)
    monkeypatch.setattr(jax_dummy.PROCESS_STATE, "preallocate_at_load", None)
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["FlaxAgents"]["class"] = "SharedFlaxModelAgent"
    setting["simulation"]["sessions"][0]["iterationSteps"] = 1
    runner = _make_runner(setting=setting)
    # the model loaded on the main process is not available on the spawned worker processes
    # without the worker initializer
    jax_dummy.load_shared_model(seed=SHARED_MODEL_SEED)
    runner._setup()
    with pytest.raises(
        RuntimeError, match="the shared model is not loaded on this process"
    ):
        runner._run()
    assert runner.executor is None


def test_jit_compiled_once_per_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["numParallel"] = 1
    runner = _make_runner(setting=setting)
    runner._setup()
    executor = runner._get_executor()
    # keep the executor after the simulation to ask the worker process
    monkeypatch.setattr(runner, "_shutdown_executor", lambda: None)
    runner._run()
    logger = runner.logger
    assert isinstance(logger, DummyLogger2)
    # 10 steps x 4 agents are asked in 10 tasks on the only worker process
    assert logger.n_order_log == 40
    assert executor.submit(jax_dummy.get_trace_count).result() == 1


def test_agents_are_picklable() -> None:
    runner = SequentialRunner(
        settings=copy.deepcopy(DEFAULT_SETTING), prng=random.Random(42)
    )
    runner.class_register(cls=jax_dummy.FlaxPriceAgent)
    runner._setup()
    agent = runner.simulator.normal_frequency_agents[0]
    assert isinstance(agent, jax_dummy.FlaxPriceAgent)
    copied_agent = pickle.loads(pickle.dumps(agent))  # nosec B301 # trusted data
    assert isinstance(copied_agent, jax_dummy.FlaxPriceAgent)
    assert copied_agent.model == agent.model
    assert jax.tree_util.tree_structure(
        copied_agent.params
    ) == jax.tree_util.tree_structure(agent.params)
    for copied_leaf, leaf in zip(
        jax.tree_util.tree_leaves(copied_agent.params),
        jax.tree_util.tree_leaves(agent.params),
    ):
        assert isinstance(copied_leaf, jax.Array)
        assert copied_leaf.dtype == leaf.dtype
        assert bool((copied_leaf == leaf).all())


def test_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_CLIENT_MEM_FRACTION",
    ]:
        monkeypatch.delenv(name, raising=False)
    runner = _make_runner(setting=DEFAULT_SETTING)
    runner._setup()
    assert runner.jax_platforms is None
    assert runner.jax_preallocate is None
    assert runner.jax_memory_fraction is None
    assert runner._get_worker_environment() == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
    }
    config = runner._get_executor().submit(jax_dummy.get_jax_worker_config).result()
    assert config["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert config["XLA_PYTHON_CLIENT_MEM_FRACTION"] is None
    assert config["jax_platforms"] == os.environ.get("JAX_PLATFORMS")
    # the main process is not configured
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    runner._shutdown_executor()


def test_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    # the new name of XLA_PYTHON_CLIENT_MEM_FRACTION cannot be used together with it
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.25")
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["jaxPlatforms"] = "cpu"
    setting["simulation"]["jaxPreallocate"] = True
    setting["simulation"]["jaxMemoryFraction"] = 0.5
    runner = _make_runner(setting=setting)
    runner._setup()
    assert runner.jax_platforms == "cpu"
    assert runner.jax_preallocate is True
    assert runner.jax_memory_fraction == 0.5
    assert runner._get_worker_environment() == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
    }
    config = runner._get_executor().submit(jax_dummy.get_jax_worker_config).result()
    assert config["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert config["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
    assert config["XLA_CLIENT_MEM_FRACTION"] is None
    assert config["jax_platforms"] == "cpu"
    assert config["default_backend"] == "cpu"
    # the main process is not configured
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    assert os.environ["XLA_CLIENT_MEM_FRACTION"] == "0.25"
    runner._shutdown_executor()


def test_config_inherited_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
    monkeypatch.delenv("XLA_CLIENT_MEM_FRACTION", raising=False)
    # the worker processes inherit the environment variables if the keys are not set
    runner = _make_runner(setting=DEFAULT_SETTING)
    runner._setup()
    assert runner._get_worker_environment() == {}
    config = runner._get_executor().submit(jax_dummy.get_jax_worker_config).result()
    assert config["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert config["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.3"
    runner._shutdown_executor()
    # the keys override the environment variables
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["jaxPreallocate"] = False
    setting["simulation"]["jaxMemoryFraction"] = 0.5
    runner = _make_runner(setting=setting)
    runner._setup()
    config = runner._get_executor().submit(jax_dummy.get_jax_worker_config).result()
    assert config["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert config["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
    runner._shutdown_executor()


def test_jax_platforms_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # JAX reads JAX_PLATFORMS when it is imported, so JAX on the worker processes fails with this
    # inherited value unless jaxPlatforms is applied to the config of JAX
    monkeypatch.setenv("JAX_PLATFORMS", "nosuchplatform")
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["jaxPlatforms"] = "cpu"
    runner = _make_runner(setting=setting)
    runner._setup()
    config = runner._get_executor().submit(jax_dummy.get_jax_worker_config).result()
    assert config["jax_platforms"] == "cpu"
    assert config["default_backend"] == "cpu"
    runner._shutdown_executor()


def test_worker_environment_set_before_jax_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # XLA_FLAGS takes effect only if it is set before JAX initializes its backends
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["jaxPlatforms"] = "cpu"
    _, runner = _make_runners(
        setting=setting, runner_class=XlaFlagsJaxAgentParallelRunner
    )
    runner._setup()
    assert runner._get_executor().submit(jax_dummy.get_device_count).result() == 2
    assert "XLA_FLAGS" not in os.environ
    runner._shutdown_executor()


@pytest.mark.parametrize("jax_memory_fraction, expected", [(1, "1.0"), (0.1, "0.1")])
def test_jax_memory_fraction(jax_memory_fraction: Any, expected: str) -> None:
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["jaxMemoryFraction"] = jax_memory_fraction
    runner = _make_runner(setting=setting)
    runner._setup()
    runner._shutdown_executor()
    assert runner._get_worker_environment()["XLA_PYTHON_CLIENT_MEM_FRACTION"] == (
        expected
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("jaxPlatforms", ""),
        ("jaxPlatforms", 1),
        ("jaxPlatforms", ["cpu"]),
        ("jaxPlatforms", None),
        ("jaxPreallocate", "false"),
        ("jaxPreallocate", 0),
        ("jaxPreallocate", None),
        ("jaxMemoryFraction", 0),
        ("jaxMemoryFraction", -0.5),
        ("jaxMemoryFraction", 1.5),
        ("jaxMemoryFraction", float("nan")),
        ("jaxMemoryFraction", "0.5"),
        ("jaxMemoryFraction", True),
        ("jaxMemoryFraction", None),
    ],
)
def test_config_invalid(key: str, value: Any) -> None:
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"][key] = value
    runner = _make_runner(setting=setting)
    with pytest.raises(ValueError, match=f"simulation.{key} must be"):
        runner._setup()
    assert runner.executor is None


def test_worker_initializer_failure() -> None:
    setting = copy.deepcopy(DEFAULT_SETTING)
    setting["simulation"]["sessions"][0]["iterationSteps"] = 1
    _, runner = _make_runners(
        setting=setting, runner_class=FailingJaxAgentParallelRunner
    )
    runner._setup()
    with pytest.raises(RuntimeError, match="the worker initializer failed") as exc_info:
        runner._run()
    assert "error in worker initializer" in str(exc_info.value.__cause__)
    assert runner.executor is None
