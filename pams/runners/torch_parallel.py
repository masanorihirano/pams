import os
import random
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


def _import_torch() -> Any:
    """Import PyTorch (internal function).

    PyTorch is an optional dependency of pams, so it is imported only when it is used.

    Returns:
        Any: :mod:`torch` module.

    """
    try:
        # this imports torch as well. It is not at the top level because PyTorch is optional.
        import torch.multiprocessing  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "TorchAgentParallelRunner requires PyTorch, but it is not installed. "
            "Install it by `pip install torch`; see https://pytorch.org/get-started/locally/ "
            "for the builds for your platform and accelerator."
        ) from e
    return torch


def _initialize_torch_worker(num_threads: int, sharing_strategy: Optional[str]) -> None:
    """Configure PyTorch on a worker process (internal function).

    This function is a module-level function so that it can be pickled and executed on the worker
    processes of :class:`pams.runners.TorchAgentParallelRunner` as their initializer.

    Args:
        num_threads (int): number of threads used by PyTorch on the worker process.
        sharing_strategy (str, Optional): sharing strategy of :mod:`torch.multiprocessing` on the
            worker process. If it is None, the default strategy is kept.

    Returns:
        None

    """
    torch = _import_torch()
    torch.set_num_threads(num_threads)
    if sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)


class TorchAgentParallelRunner(MultiProcessAgentParallelRunner):
    """Multi Process Agent Parallel runner class for agents using PyTorch. This is experimental.

    This runner is :class:`pams.runners.MultiProcessAgentParallelRunner` configured for agents
    whose :func:`pams.agents.Agent.submit_orders` uses `PyTorch <https://pytorch.org/>`_, e.g., to
    predict the order price by a neural network model. The simulation results are identical to
    those of :class:`pams.runners.SequentialRunner` with the same settings and the same seed as
    long as the agents are deterministic (see below).

    PyTorch is an optional dependency of pams. It is imported when this runner is created, and an
    ImportError is raised if it is not installed.

    In addition to :class:`pams.runners.MultiProcessAgentParallelRunner`, this runner

    - starts the worker processes by ``spawn`` by default (:attr:`default_start_method`), which
      can be changed by ``simulation.startMethod``. ``fork`` is unsafe for PyTorch: CUDA cannot be
      initialized in a forked process, and only the forking thread is copied to a forked process,
      so a worker forked after PyTorch has started its thread pools on the main process can
      deadlock. Because of ``spawn``, the code running this runner must be under
      ``if __name__ == "__main__":`` on every platform, including Linux.
    - sets the number of threads of PyTorch on each worker process by
      :func:`torch.set_num_threads` to ``simulation.torchNumThreads``. By default, PyTorch uses
      about as many threads as the CPU cores in every process, so ``numParallel`` workers would
      oversubscribe the CPUs. The default is the number of threads of PyTorch on the main process,
      :func:`torch.get_num_threads`, divided by the number of workers that can run tasks at the
      same time, i.e., the smaller of ``numParallel`` and the largest ``maxNormalOrders`` of the
      sessions (at least one).
    - sets the sharing strategy of :mod:`torch.multiprocessing` to
      :attr:`torch_sharing_strategy`, i.e., ``file_system``, on the main process when it is set up
      and on each worker process (see below).

    Importing PyTorch registers the reductions of tensors to
    :class:`multiprocessing.reduction.ForkingPickler`, with which
    :class:`concurrent.futures.ProcessPoolExecutor` pickles the tasks. Therefore, the tensors held
    by agents, e.g., the parameters of their models, are moved to shared memory when they are sent
    to a worker process for the first time, and each task sends only their handles instead of
    their data. This also happens with :class:`pams.runners.MultiProcessAgentParallelRunner`, but
    on Linux, the default sharing strategy, ``file_descriptor``, duplicates a file descriptor for
    each tensor in each task, because every task pickles the tensors of all the agents. It is
    several times slower, and it fails, e.g., with "Too many open files", once the tensors of the
    tasks in flight exceed the limit of open files (1024 by default on many Linux systems, which 50
    agents with 6 tensors each already exceed with ``numParallel`` 4). The ``file_system``
    strategy, the only one on Windows and macOS, sends the name of the shared memory instead.
    Still, every tensor costs opening the shared memory in every task, and the shared memory,
    i.e., ``/dev/shm`` on Linux, must be large enough for the tensors of all the agents, e.g., by
    ``--shm-size`` of Docker, whose default is 64 MB.

    The worker processes see the same memory as the main process, so in-place modifications of the
    tensors in :func:`pams.agents.Agent.submit_orders`, e.g., by training a model, change the
    tensors on the main process, unlike other modifications of the agents on the worker processes,
    which are discarded. Keep the tensors read-only in :func:`pams.agents.Agent.submit_orders`.

    Subclasses can override ``_get_worker_initializer`` and ``_get_worker_initargs``, e.g., to load
    a model once per worker process. In that case, the initializer should call
    ``_initialize_torch_worker`` with the arguments given by this class, too.

    .. note::
        For the results identical to :class:`pams.runners.SequentialRunner`, agents must derive
        their randomness from their own pseudo random number generator ``prng``, e.g., by
        ``torch.Generator().manual_seed(self.prng.randrange(2**32))``, because only its state is
        synchronized from the worker processes. The global random number generator of PyTorch on
        each worker process is independent of the main process and of the other workers. Also,
        the results of some operations of PyTorch, e.g., matrix products, can differ in rounding
        with the number of threads and with the device, and such a difference can change the
        course of the simulation. To compare with :class:`pams.runners.SequentialRunner`, run it
        after ``torch.set_num_threads(n)`` and set ``simulation.torchNumThreads`` to ``n``.

    .. note::
        This runner does not choose the device. The worker processes see the same devices as the
        main process, and agents choose the device in :func:`pams.agents.Agent.submit_orders`.
        Every worker process using CUDA creates its own CUDA context, which takes hundreds of
        megabytes of the GPU memory. CUDA tensors held by agents would be sent by CUDA IPC, which
        is not supported on Windows, so keep the tensors held by agents on the CPU. Moving them to
        the GPU in :func:`pams.agents.Agent.submit_orders` copies them in every task, because
        each task gets new copies of the agents. To copy a model to the GPU only once per worker
        process, load it by the worker initializer into a module-level variable and use it in
        :func:`pams.agents.Agent.submit_orders`.
    """

    #: Optional[str]: start method of the worker processes used when ``simulation.startMethod`` is
    #: not set in the config. ``spawn`` is used because ``fork`` is unsafe for PyTorch.
    default_start_method: Optional[str] = "spawn"

    #: Optional[str]: sharing strategy of :mod:`torch.multiprocessing` set on the main process and
    #: on the worker processes. It must be one of
    #: :func:`torch.multiprocessing.get_all_sharing_strategies`. None keeps the current strategy.
    torch_sharing_strategy: Optional[str] = "file_system"

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
        _import_torch()
        super().__init__(settings, prng, logger, simulator_class)
        #: Optional[int]: number of threads of PyTorch on each worker process given by
        #: ``simulation.torchNumThreads``. None means the default, i.e., the number of threads of
        #: PyTorch on the main process divided by the smaller of ``num_parallel`` and the largest
        #: ``max_normal_orders`` of the sessions (at least one).
        self.torch_num_threads: Optional[int] = None

    def _setup(self) -> None:
        """Set up the simulation (internal method).

        In addition to :func:`pams.runners.MultiProcessAgentParallelRunner._setup`,
        ``simulation.torchNumThreads`` is read and the sharing strategy of
        :mod:`torch.multiprocessing` is set before the executor is prepared.
        """
        if (
            "simulation" in self.settings
            and "torchNumThreads" in self.settings["simulation"]
        ):
            torch_num_threads = self.settings["simulation"]["torchNumThreads"]
            if (
                isinstance(torch_num_threads, bool)
                or not isinstance(torch_num_threads, int)
                or torch_num_threads < 1
            ):
                raise ValueError(
                    "simulation.torchNumThreads must be a positive integer, "
                    f"but {torch_num_threads} is given"
                )
            self.torch_num_threads = torch_num_threads
        if self.torch_sharing_strategy is not None:
            _import_torch().multiprocessing.set_sharing_strategy(
                self.torch_sharing_strategy
            )
        super()._setup()

    def _get_worker_initializer(self) -> Optional[Callable[..., Any]]:
        """Get the function called once on each worker before it runs any task (internal method).

        Returns:
            Callable[..., Any], Optional: initializer of the workers, which imports PyTorch and sets
            its number of threads and its sharing strategy.

        """
        return _initialize_torch_worker

    def _get_worker_initargs(self) -> Tuple[Any, ...]:
        """Get the arguments of the worker initializer (internal method).

        Returns:
            Tuple[Any, ...]: the number of threads of PyTorch on each worker process, i.e.,
            :attr:`torch_num_threads`, or the number of threads of PyTorch on the main process
            divided by the number of workers that can run tasks at the same time (at least one)
            if it is None, and :attr:`torch_sharing_strategy`.

        """
        num_threads: int
        if self.torch_num_threads is not None:
            num_threads = self.torch_num_threads
        else:
            # at most max_normal_orders agents are asked at the same time, so the other workers
            # stay idle and do not need threads.
            max_normal_orders: int = max(
                (session.max_normal_orders for session in self.simulator.sessions),
                default=self.num_parallel,
            )
            num_busy_workers: int = max(min(self.num_parallel, max_normal_orders), 1)
            num_threads = max(_import_torch().get_num_threads() // num_busy_workers, 1)
        return (num_threads, self.torch_sharing_strategy)
