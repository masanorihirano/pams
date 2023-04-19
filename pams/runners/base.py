import json
import os
import random
import time
from abc import ABC
from abc import abstractmethod
from io import TextIOBase
from io import TextIOWrapper
from typing import Dict
from typing import List
from typing import Optional
from typing import TextIO
from typing import Type
from typing import Union

from ..agents import Agent
from ..agents import HighFrequencyAgent
from ..logs.base import Logger
from ..simulator import Simulator


class Runner(ABC):
    """Runner of the market simulator class (Abstract class).

    .. seealso::
        - :class:`pams.runners.SequentialRunner`
    """

    def __init__(
        self,
        settings: Union[Dict, TextIOBase, TextIO, TextIOWrapper, os.PathLike, str],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator,
    ):
        """initialize.

        Args:
            settings (Union[Dict, TextIOBase, TextIO, TextIOWrapper, os.PathLike, str]): runner configuration.
                You can set python dictionary, a file pointer, or a file path.
            prng (random.Random, Optional): pseudo random number generator for this runner.
            logger (Logger, Optional): logger instance.
            simulator_class (Type[Simulator]): type of simulator.

        Returns:
            None
        """
        self.settings: Dict
        if isinstance(settings, Dict):
            self.settings = settings
        elif (
            isinstance(settings, TextIOBase)
            or isinstance(settings, TextIO)
            or isinstance(settings, TextIOWrapper)
        ):
            self.settings = json.load(fp=settings)
        else:
            self.settings = json.load(fp=open(settings, mode="r"))
        self._prng: random.Random = prng if prng is not None else random.Random()
        self.logger = logger
        self.simulator: Simulator = simulator_class(
            prng=random.Random(self._prng.randint(0, 2**31))
        )
        self.registered_classes: List[Type] = []

    def main(self) -> None:
        """main process. The process is executed while measuring time."""
        setup_start_time_ns = time.time_ns()
        self._setup()
        start_time_ns = time.time_ns()
        self._run()
        end_time_ns = time.time_ns()
        print(
            "# INITIALIZATION TIME " + str((start_time_ns - setup_start_time_ns) / 1e9)
        )
        print("# EXECUTION TIME " + str((end_time_ns - start_time_ns) / 1e9))

    def class_register(self, cls: Type) -> None:
        """register class. This method is used for user-defined classes.

        Usually, user-defined classes, i.e., the classes you implemented for your original simulation, cannot be referred from
        pams package, especially from simulation runners. Therefore, the class registration to the runner is necessary.

        Args:
            cls (Type): class to register.

        Returns:
            None
        """
        self.registered_classes.append(cls)

    @abstractmethod
    def _setup(self) -> None:
        """internal usage class for setup. This method should be implemented in descendants."""
        pass

    @abstractmethod
    def _run(self) -> None:
        """internal usage class for simulation running. This method should be implemented in descendants.

        Usually the process in this methods should be control simulation flow and parallelization.
        """
        pass

    @staticmethod
    def judge_hft_or_not(agent: Agent) -> bool:
        """determine if the agent is type of the :class:`pams.agents.HighFrequencyAgent`.

        Args:
            agent (Agent): agent instance.

        Returns:
            bool: whether the agent class is the :class:`pams.agents.HighFrequencyAgent` or not.
        """
        return isinstance(agent, HighFrequencyAgent)
