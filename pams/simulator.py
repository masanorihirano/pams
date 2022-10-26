import random
from typing import Dict
from typing import List
from typing import Optional

from .agent import Agent
from .events.base import EventABC
from .events.base import EventHook
from .logs import Logger
from .market import Market


class Simulator:
    def __init__(self, prng: random.Random, logger: Optional[Logger] = None) -> None:
        self._prng = prng
        self.logger: Optional[Logger] = logger
        self.n_events: int = 0
        self.events: List[EventABC] = []
        self.id2event: Dict[int, EventABC] = {}
        self.event_hooks: List[EventHook] = []
        self.events_dict: Dict[str, Dict[int, List[EventHook]]] = {
            "simulator_before": {},
            "simulator_after": {},
            "market_before": {},
            "market_after": {},
            "agent_before": {},
            "agent_after": {},
        }

        self.n_agents: int = 0
        self.agents: List[Agent] = []
        self.high_frequency_agents: List[Agent] = []
        self.id2agent: Dict[int, Agent] = {}

        self.n_markets: int = 0
        self.markets: List[Market] = []
        self.id2market: Dict[int, Market]

        # ToDo fundamentals

    # ToDo get_xxx_by_name
