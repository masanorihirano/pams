import os
import random
from concurrent.futures import Executor
from io import TextIOWrapper
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
from .agent_parallel import _initialize_worker


def _import_jax() -> Any:
    """Import JAX (internal function).

    JAX is an optional dependency of pams, so it is imported only when it is used.

    Returns:
        Any: the ``jax`` module.

    """
    try:
        import jax
    except ImportError as e:
        raise ImportError(
            "JaxAgentParallelRunner requires JAX, which is not installed. Install it, e.g., by"
            ' `pip install jax flax` for CPUs or `pip install "jax[cuda12]" flax` for NVIDIA'
            " GPUs (flax is needed only for Flax models). See"
            " https://docs.jax.dev/en/latest/installation.html for the details."
        ) from e
    return jax


def _initialize_jax_worker(
    environment: Dict[str, str],
    platforms: Optional[str],
    initializer: Optional[Callable[..., Any]],
    initargs: Tuple[Any, ...],
) -> None:
    """Configure JAX on a worker process and call the worker initializer (internal function).

    The environment variables are set before JAX initializes its backends, which happens when JAX
    is used for the first time on the process, e.g., when a jax array is created. Therefore, JAX
    must not be used when the modules imported by the worker process, including the main module,
    are imported.

    Args:
        environment (Dict[str, str]): environment variables set on the worker process.
        platforms (str, Optional): platforms of JAX, i.e., ``jax_platforms``, e.g., ``"cpu"``. If it
            is None, it is not changed.
        initializer (Callable[..., Any], Optional): worker initializer called after JAX is
            configured. If it is None, nothing is called.
        initargs (Tuple[Any, ...]): arguments of the worker initializer.

    Returns:
        None

    """
    os.environ.update(environment)
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in environment:
        # recent versions of JAX raise an error if the new name is also set
        os.environ.pop("XLA_CLIENT_MEM_FRACTION", None)
    jax = _import_jax()
    if platforms is not None:
        # jax_platforms is read from JAX_PLATFORMS when jax is imported, which may have happened
        # already, e.g., when the main module was imported.
        jax.config.update("jax_platforms", platforms)
    if initializer is not None:
        initializer(*initargs)


class JaxAgentParallelRunner(MultiProcessAgentParallelRunner):
    """JAX Agent Parallel runner class. This is experimental.

    This runner is :class:`pams.runners.MultiProcessAgentParallelRunner` for agents whose
    :func:`pams.agents.Agent.submit_orders` uses `JAX <https://github.com/jax-ml/jax>`_, e.g.,
    neural network models of `Flax <https://github.com/google/flax>`_. The simulation results are
    identical to those of :class:`pams.runners.SequentialRunner` with the same settings and the same
    seed if the agents are deterministic given their pseudo random number generators.

    JAX is an optional dependency of pams. It is imported when this runner is created, and an
    ImportError is raised if it is not installed. ``import pams`` does not import it.

    The worker processes are started by ``spawn`` by default, i.e., :attr:`default_start_method` is
    ``"spawn"``, because ``fork`` is unsafe for JAX. Once JAX is initialized on the main process,
    e.g., when the agents create their models, JAX runs its own threads. A forked worker process has
    only a copy of the thread calling ``fork``, so a lock held by another thread is never released
    and the worker process can deadlock. JAX warns about it when ``fork`` is called.
    ``simulation.startMethod`` in the config still takes precedence.

    Each worker process configures JAX once before it runs any task, by the following keys in
    ``simulation`` of the config:

    - ``jaxPlatforms`` (str, Optional): platforms that JAX uses on the worker processes, e.g.,
      ``"cpu"`` or ``"cuda"``, i.e., ``jax_platforms`` of JAX. If it is not set, ``JAX_PLATFORMS``
      in the environment is used if it is set; otherwise, JAX chooses the platform.
    - ``jaxPreallocate`` (bool): whether JAX preallocates the memory of GPUs on each worker process,
      i.e., ``XLA_PYTHON_CLIENT_PREALLOCATE``. The default is false, so that several worker
      processes can share a GPU, because JAX preallocates 75% of the memory of the GPU by default.
    - ``jaxMemoryFraction`` (float, Optional): fraction of the memory of the GPU that JAX on each
      worker process can use, in (0, 1], i.e., ``XLA_PYTHON_CLIENT_MEM_FRACTION``. If it is not set,
      the default of JAX (0.75) is used.

    Then, the initializer given by ``_get_worker_initializer`` is called as in
    :class:`pams.runners.MultiProcessAgentParallelRunner`, e.g., to load a read-only model once per
    worker process. The environment variables set on the worker processes are given by
    ``_get_worker_environment``, which subclasses can override.

    .. note::
        The worker processes persist during the simulation. Thus, a function compiled by
        :func:`jax.jit` is compiled once per worker process (for each shape and type of the
        arguments) at its first call, and the compiled function is reused in the following steps.
        The functions compiled by :func:`jax.jit` cannot be pickled, so define them at the top level
        of a module instead of keeping them as attributes of the agents.

    .. note::
        The agents are pickled and copied to a worker process for each task, including their jax
        arrays and Flax modules and parameters, which are copied to the default device of the worker
        process. Large models should be loaded once per worker process by the worker initializer
        instead (see :class:`pams.runners.MultiProcessAgentParallelRunner`). Only the state of
        ``agent.prng`` is sent back to the main process, so the agents should derive their random
        numbers, including the keys of :mod:`jax.random`, from ``agent.prng``.

    .. note::
        Do not use JAX at the top level of the modules imported by the worker processes, such as the
        main module, and do not pass jax arrays to the worker initializer; otherwise, JAX is
        initialized before it is configured, and the settings above are ignored.
    """

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
        _import_jax()
        super().__init__(settings, prng, logger, simulator_class)
        self.jax_platforms: Optional[str] = None
        self.jax_preallocate: bool = False
        self.jax_memory_fraction: Optional[float] = None

    def _setup(self) -> None:
        """Set up the simulation (internal method).

        In addition to :func:`pams.runners.MultiProcessAgentParallelRunner._setup`,
        ``simulation.jaxPlatforms``, ``simulation.jaxPreallocate`` and
        ``simulation.jaxMemoryFraction`` are read before the executor is prepared.
        """
        simulation_settings: Dict = self.settings.get("simulation", {})
        if "jaxPlatforms" in simulation_settings:
            jax_platforms = simulation_settings["jaxPlatforms"]
            if not isinstance(jax_platforms, str) or jax_platforms == "":
                raise ValueError(
                    "simulation.jaxPlatforms must be a non-empty string, "
                    f"but {jax_platforms} is given"
                )
            self.jax_platforms = jax_platforms
        if "jaxPreallocate" in simulation_settings:
            jax_preallocate = simulation_settings["jaxPreallocate"]
            if not isinstance(jax_preallocate, bool):
                raise ValueError(
                    "simulation.jaxPreallocate must be a boolean, "
                    f"but {jax_preallocate} is given"
                )
            self.jax_preallocate = jax_preallocate
        if "jaxMemoryFraction" in simulation_settings:
            jax_memory_fraction = simulation_settings["jaxMemoryFraction"]
            if (
                isinstance(jax_memory_fraction, bool)
                or not isinstance(jax_memory_fraction, (int, float))
                or not 0 < jax_memory_fraction <= 1
            ):
                raise ValueError(
                    "simulation.jaxMemoryFraction must be a number in (0, 1], "
                    f"but {jax_memory_fraction} is given"
                )
            self.jax_memory_fraction = float(jax_memory_fraction)
        super()._setup()

    def _get_worker_environment(self) -> Dict[str, str]:
        """Get the environment variables set on each worker process (internal method).

        The variables are set before JAX is initialized on the worker process. Subclasses can
        override this method to set other variables, e.g., ``XLA_FLAGS``.

        Returns:
            Dict[str, str]: environment variables. ``XLA_PYTHON_CLIENT_PREALLOCATE`` is always set
            by ``simulation.jaxPreallocate``, and ``XLA_PYTHON_CLIENT_MEM_FRACTION`` is set if
            ``simulation.jaxMemoryFraction`` is set.

        """
        environment: Dict[str, str] = {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true" if self.jax_preallocate else "false"
        }
        if self.jax_memory_fraction is not None:
            environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
                self.jax_memory_fraction
            )
        return environment

    def _create_executor(self) -> Executor:
        """Create a new executor (internal method).

        In addition to :func:`pams.runners.MultiProcessAgentParallelRunner._create_executor`, each
        worker process configures JAX by the environment variables given by
        ``_get_worker_environment`` and ``simulation.jaxPlatforms`` before it calls the initializer
        given by ``_get_worker_initializer`` and ``_get_worker_initargs``.

        Returns:
            Executor: new executor.

        """
        return self._parallel_pool_provider(
            max_workers=self.num_parallel,
            mp_context=self._get_mp_context(),
            initializer=_initialize_worker,
            initargs=(
                _initialize_jax_worker,
                (
                    self._get_worker_environment(),
                    self.jax_platforms,
                    self._get_worker_initializer(),
                    self._get_worker_initargs(),
                ),
            ),
        )
