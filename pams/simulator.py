import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from .agent import Agent
from .events.base import EventABC
from .events.base import EventHook
from .fundamentals import Fundamentals
from .logs import Logger
from .market import Market


class Simulator:
    def __init__(
        self,
        prng: random.Random,
        logger: Optional[Logger] = None,
        fundamental_class: Type[Fundamentals] = Fundamentals,
    ) -> None:
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

        self.fundamentals = fundamental_class(
            prng=random.Random(self._prng.randint(0, 2**31))
        )

    def _add_market(self, market: Market):
        # ToDo
        pass

    def _update_time_on_market(self, market: Market) -> None:
        market._update_time(
            next_fundamental_price=self.fundamentals.get_fundamental_price(
                market_id=market.market_id, time=market.get_time() + 1
            )
        )

    def _update_times_on_markets(self, markets: List[Market]) -> None:
        for market in markets:
            self._update_time_on_market(market=market)

    # ToDo get_xxx_by_name
