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
from .high_frequency_agent import HighFrequencyAgent
from .logs import Logger
from .market import Market
from .session import Session


class Simulator:
    def __init__(
        self,
        prng: random.Random,
        logger: Optional[Logger] = None,
        fundamental_class: Type[Fundamentals] = Fundamentals,
    ) -> None:
        self._prng = prng
        self.logger: Optional[Logger] = logger

        # ToDo: move to session?
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
        self.name2event: Dict[str, EventHook] = {}

        self.n_agents: int = 0
        self.agents: List[Agent] = []
        self.high_frequency_agents: List[Agent] = []
        self.normal_frequency_agents: List[Agent] = []
        self.id2agent: Dict[int, Agent] = {}
        self.name2agent: Dict[str, Agent] = {}
        self.agents_group_name2agent: Dict[str, List[Agent]] = {}

        self.n_markets: int = 0
        self.markets: List[Market] = []
        self.id2market: Dict[int, Market] = {}
        self.name2market: Dict[str, Market] = {}
        self.markets_group_name2market: Dict[str, List[Market]] = {}

        self.fundamentals = fundamental_class(
            prng=random.Random(self._prng.randint(0, 2**31))
        )

        self.n_sessions: int = 0
        self.sessions: List[Session] = []
        self.id2session: Dict[int, Session] = {}
        self.name2session: Dict[str, Session] = {}

    def _add_market(self, market: Market, group_name: Optional[str] = None) -> None:
        if market in self.markets:
            raise ValueError("market is already registered")
        if market.market_id in self.id2market:
            raise ValueError(f"market_id {market.market_id} is duplicated")
        if market.name in self.name2market:
            raise ValueError(f"market name {market.name} is duplicate")
        self.markets.append(market)
        self.n_markets += 1
        self.id2market[market.market_id] = market
        self.name2market[market.name] = market
        if group_name is not None:
            if group_name not in self.markets_group_name2market:
                self.markets_group_name2market[group_name] = []
            self.markets_group_name2market[group_name].append(market)

    def _add_agent(self, agent: Agent, group_name: Optional[str] = None) -> None:
        if agent in self.agents:
            raise ValueError("agent is already registered")
        if agent.agent_id in self.id2agent:
            raise ValueError(f"agent_id {agent.agent_id} is duplicated")
        if agent.name in self.name2agent:
            raise ValueError(f"agent name {agent.name} is duplicate")
        self.agents.append(agent)
        self.n_agents += 1
        self.id2agent[agent.agent_id] = agent
        self.name2agent[agent.name] = agent
        if isinstance(agent, HighFrequencyAgent):
            self.high_frequency_agents.append(agent)
        else:
            self.normal_frequency_agents.append(agent)
        if group_name is not None:
            if group_name not in self.agents_group_name2agent:
                self.agents_group_name2agent[group_name] = []
            self.agents_group_name2agent[group_name].append(agent)

    def _add_session(self, session: Session) -> None:
        if session in self.sessions:
            raise ValueError("session is already registered")
        if session.session_id in self.id2session:
            raise ValueError(f"session_id {session.session_id} is duplicated")
        if session.name in self.name2session:
            raise ValueError(f"session name {session.name} is duplicate")
        self.sessions.append(session)
        self.n_sessions += 1
        self.id2session[session.session_id] = session
        self.name2session[session.name] = session

    def _update_time_on_market(self, market: Market) -> None:
        market._update_time(
            next_fundamental_price=self.fundamentals.get_fundamental_price(
                market_id=market.market_id, time=market.get_time() + 1
            )
        )

    def _update_times_on_markets(self, markets: List[Market]) -> None:
        for market in markets:
            self._update_time_on_market(market=market)

    def trigger_event_before_order(self, order: "Order") -> None:  # type: ignore
        # ToDO
        pass

    def trigger_event_after_order(self, order_log: "OrderLog") -> None:  # type: ignore
        # ToDO
        pass

    def trigger_event_after_execution(self, execution_log: "ExecutionLog") -> None:  # type: ignore
        # ToDO
        pass

    def trigger_event_before_session(self, session: "Session") -> None:  # type: ignore
        # ToDO
        pass

    def trigger_event_after_session(self, session: "Session") -> None:  # type: ignore
        # ToDO
        pass

    def trigger_event_before_step_for_market(self, market: "Market") -> None:  # type: ignore
        # ToDO
        pass

    def trigger_event_after_step_for_market(self, market: "Market") -> None:  # type: ignore
        # ToDO
        pass

    # ToDo get_xxx_by_name
