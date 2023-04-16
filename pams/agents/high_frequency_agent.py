from abc import ABC

from .base import Agent


class HighFrequencyAgent(Agent, ABC):
    """High-frequency trading Agent class.

    High-frequency traders are treated as a special agent.

    This class inherits from the :class:`pams.agents.Agent` class and ABC class.

    .. seealso:
      -  :class:`pams.agents.ArbitrageAgent` for example.
    """

    pass
