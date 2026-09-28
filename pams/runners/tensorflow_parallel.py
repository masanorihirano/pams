import os
import random
from io import TextIOWrapper
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from ..logs.base import Logger
from ..simulator import Simulator
from .agent_parallel import MultiProcessAgentParallelRunner


def _import_tensorflow() -> ModuleType:
    """Import TensorFlow (internal function).

    TensorFlow is an optional dependency of PAMS, so it is imported only when it is used.

    Returns:
        ModuleType: the ``tensorflow`` module.

    """
    try:
        import tensorflow
    except ImportError as e:
        raise ImportError(
            "TensorFlowAgentParallelRunner requires TensorFlow, which is not installed."
            " Install it, e.g., by `pip install tensorflow`"
            " (or `pip install tensorflow-cpu` for the CPU-only build)."
        ) from e
    return tensorflow


def _initialize_tensorflow_worker(
    intra_op_parallelism_threads: int,
    inter_op_parallelism_threads: int,
    gpu_memory_growth: bool,
) -> None:
    """Configure TensorFlow on a worker process (internal function).

    This function is the worker initializer of
    :class:`pams.runners.TensorFlowAgentParallelRunner`. It is called once on each worker process
    before the process runs any task, i.e., before TensorFlow is initialized on the process.

    Args:
        intra_op_parallelism_threads (int): number of threads used to run one operation.
        inter_op_parallelism_threads (int): number of threads used to run independent operations.
        gpu_memory_growth (bool): whether to enable memory growth for all the visible GPUs.

    Returns:
        None

    """
    tf = _import_tensorflow()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            intra_op_parallelism_threads
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            inter_op_parallelism_threads
        )
        if gpu_memory_growth:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        raise RuntimeError(
            "TensorFlow cannot be configured because it is already initialized on this worker"
            " process. This happens if the worker process is started by fork after TensorFlow is"
            " used on the main process, or if TensorFlow is used at the top level of the main"
            " module, which the worker processes import when they are started by spawn."
        ) from e


def _check_num_threads(key: str, value: Any) -> int:
    """Check the number of threads given in the config (internal function).

    Args:
        key (str): key of the number of threads in ``simulation``.
        value (Any): value given in the config.

    Returns:
        int: the number of threads.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"simulation.{key} must be a positive integer, but {value} is given"
        )
    return value


class TensorFlowAgentParallelRunner(MultiProcessAgentParallelRunner):
    """Multi Process Agent Parallel runner class for agents using TensorFlow. This is experimental.

    This runner is :class:`pams.runners.MultiProcessAgentParallelRunner` whose worker processes are
    prepared for agents that use `TensorFlow <https://www.tensorflow.org/>`_ in
    :func:`pams.agents.Agent.submit_orders`. The simulation results are identical to those of
    :class:`pams.runners.SequentialRunner` with the same settings and the same seed as long as the
    agents are deterministic (see below).

    TensorFlow is an optional dependency of PAMS: ``import pams`` does not import it, and this
    runner imports it when it is created. If TensorFlow is not installed, an ImportError is raised.
    Install it by ``pip install tensorflow`` (or ``pip install tensorflow-cpu`` for the CPU-only
    build).

    The worker processes are started by ``spawn`` by default, because TensorFlow is not fork-safe.
    Once TensorFlow runs an operation, the process has thread pools (and CUDA contexts for GPUs),
    but a process started by ``fork`` has only a copy of the thread calling fork. In the forked
    process, operations that use the thread pools of TensorFlow hang, GPUs cannot be used, and
    TensorFlow cannot be configured anymore. Such a fork happens easily, e.g., when agents build
    their models on the main process or a :class:`pams.runners.SequentialRunner` is run before on
    the same process. A spawned process starts a new interpreter and initializes TensorFlow by
    itself. ``simulation.startMethod`` can still select another start method: ``forkserver`` is also
    safe if TensorFlow is not used at the top level of the main module, and ``fork`` works only if
    TensorFlow is not used on the main process before the simulation.

    Each worker process is configured once before it runs any task, by the following keys of
    ``simulation`` in the config:

    - ``tensorflowIntraOpThreads`` (int ≥ 1): number of threads that TensorFlow uses to run one
      operation, e.g., a matrix multiplication, in each worker process. The default is the number
      of CPUs divided by ``numParallel`` (at least 1), so that the worker processes do not use more
      threads than the CPUs in total. TensorFlow uses all the CPUs in each process by default.
    - ``tensorflowInterOpThreads`` (int ≥ 1): number of threads that TensorFlow uses to run
      independent operations at the same time in each worker process. The default is 1.
    - ``tensorflowGpuMemoryGrowth`` (bool): whether to enable memory growth for all the visible GPUs
      in each worker process. The default is true. Without memory growth, TensorFlow allocates
      almost all the memory of a GPU in the first process using it, so the other worker processes
      fail with out of memory errors when they share the GPU. Set it to false to keep the default
      behavior of TensorFlow, e.g., when each worker process uses its own GPU.

    .. note::
        As :class:`pams.runners.MultiProcessAgentParallelRunner`, this runner pickles the agents and
        the markets, i.e., the whole simulation, for each task. ``tf.Tensor`` and
        ``tf.Variable`` held by agents are pickled as copies of their values, and Keras 3
        models are pickled by saving them in the ``.keras`` format, which takes about 10 to 20 ms
        each way even for a small model, in every task. Objects made by ``tf.function`` cannot be
        pickled. Therefore, agents should not hold models as their attributes. Instead, an agent
        should hold a key of its model, such as the path of a saved model, and get the model from
        a cache at the top level of a module, e.g., a function decorated by
        :func:`functools.lru_cache` that loads the model. Then, each worker process loads the model
        once, at its first task, and reuses it in the later tasks.

    .. note::
        Only the orders and the state of the agent's pseudo random number generator (``prng``) are
        returned from the worker processes. The random states of TensorFlow, e.g., the global seed
        and ``tf.random.Generator``, are not returned, and which worker process runs an agent
        changes from run to run. For the same results as :class:`pams.runners.SequentialRunner`,
        agents should draw random numbers from ``prng``, e.g., the seeds of stateless random
        operations such as ``tf.random.stateless_normal``. Models must also give the same
        outputs on every process: TensorFlow on CPUs is deterministic in general, but some
        operations on GPUs are not unless ``tf.config.experimental.enable_op_determinism`` is
        called on the main process and in the worker processes.

    .. warning::
        This runner makes the simulation faster only if the TensorFlow computations in
        :func:`pams.agents.Agent.submit_orders` are much heavier than the cost of pickling the
        simulation in every task. Moreover, each worker process imports and initializes TensorFlow
        at its start, which takes seconds and hundreds of megabytes of memory per process.
    """

    #: Optional[str]: start method of the worker processes used when ``simulation.startMethod`` is
    #: not set in the config. TensorFlow is not fork-safe, so it is ``"spawn"``.
    default_start_method: Optional[str] = "spawn"

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
        _import_tensorflow()
        super().__init__(settings, prng, logger, simulator_class)
        self.intra_op_parallelism_threads: Optional[int] = None
        self.inter_op_parallelism_threads: int = 1
        self.gpu_memory_growth: bool = True

    def _setup(self) -> None:
        """Set up the simulation (internal method).

        In addition to :func:`pams.runners.MultiProcessAgentParallelRunner._setup`,
        ``simulation.tensorflowIntraOpThreads``, ``simulation.tensorflowInterOpThreads`` and
        ``simulation.tensorflowGpuMemoryGrowth`` are read before the executor is prepared.
        """
        if "simulation" in self.settings:
            simulation_settings: Dict = self.settings["simulation"]
            if "tensorflowIntraOpThreads" in simulation_settings:
                self.intra_op_parallelism_threads = _check_num_threads(
                    key="tensorflowIntraOpThreads",
                    value=simulation_settings["tensorflowIntraOpThreads"],
                )
            if "tensorflowInterOpThreads" in simulation_settings:
                self.inter_op_parallelism_threads = _check_num_threads(
                    key="tensorflowInterOpThreads",
                    value=simulation_settings["tensorflowInterOpThreads"],
                )
            if "tensorflowGpuMemoryGrowth" in simulation_settings:
                gpu_memory_growth = simulation_settings["tensorflowGpuMemoryGrowth"]
                if not isinstance(gpu_memory_growth, bool):
                    raise ValueError(
                        "simulation.tensorflowGpuMemoryGrowth must be a boolean, "
                        f"but {gpu_memory_growth} is given"
                    )
                self.gpu_memory_growth = gpu_memory_growth
        super()._setup()

    def _get_worker_initializer(self) -> Optional[Callable[..., Any]]:
        """Get the function called once on each worker before it runs any task (internal method).

        Returns:
            Callable[..., Any], Optional: ``_initialize_tensorflow_worker``, which configures
            TensorFlow on each worker process.

        """
        return _initialize_tensorflow_worker

    def _get_worker_initargs(self) -> Tuple[Any, ...]:
        """Get the arguments of the worker initializer (internal method).

        Returns:
            Tuple[Any, ...]: the number of intra-op threads, the number of inter-op threads, and
            whether to enable memory growth for GPUs. If ``intra_op_parallelism_threads`` is None,
            the number of CPUs divided by ``num_parallel`` (at least 1) is used as the number of
            intra-op threads.

        """
        intra_op_parallelism_threads: int = (
            self.intra_op_parallelism_threads
            if self.intra_op_parallelism_threads is not None
            else max((os.cpu_count() or 1) // self.num_parallel, 1)
        )
        return (
            intra_op_parallelism_threads,
            self.inter_op_parallelism_threads,
            self.gpu_memory_growth,
        )
