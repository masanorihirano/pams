import os
import random
from io import TextIOWrapper
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from ..agent import Agent
from ..logs import Logger
from ..market import Market
from ..simulator import Simulator
from ..utils.class_finder import find_class
from ..utils.json_extends import json_extends
from .base import Runner


class SequentialRunner(Runner):
    def __init__(
        self,
        settings: Union[Dict, TextIOWrapper, os.PathLike, str],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator,
    ):
        super().__init__(settings, prng, logger, simulator_class)
        self._pending_setups: List[Tuple[Callable, Dict]] = []

    def _generate_markets(self, market_type_names: List[str]) -> None:
        i_market = 0
        for name in market_type_names:
            market_settings: Dict = self.settings[name]
            n_markets = 1
            id_from = 0
            id_to = 0
            if "numMarkets" in market_settings:
                n_markets = int(market_settings["numMarkets"])
                id_to = n_markets - 1
            if "from" in market_settings or "to" in market_settings:
                if "from" not in market_settings or "to" not in market_settings:
                    raise ValueError(
                        f"both {name}.from and {name}.to are required in json file if you use"
                    )
                if "numMarkets" in market_settings:
                    raise ValueError(
                        f"{name}.numMarkets and ({name}.from or {name}.to) cannot be used at the same time"
                    )
                n_markets = market_settings["to"] - market_settings["from"]
                id_from = market_settings["from"]
                id_to = market_settings["to"]
            prefix: str
            if "prefix" in market_settings:
                prefix = market_settings["prefix"]
            else:
                prefix = name + ("-" if n_markets > 1 else "")
            market_settings = json_extends(
                whole_json=self.settings,
                parent_name=name,
                target_json=market_settings,
                excludes_fields=["numMarkets", "from", "to", "prefix"],
            )
            if "class" not in market_settings:
                raise ValueError(f"class is not defined for {name}")
            market_class: Type[Market] = find_class(name=market_settings["class"])
            if not issubclass(market_class, Market):
                raise ValueError(
                    f"market class for {name} does not inherit Market class"
                )
            if "fundamentalPrice" in market_settings:
                fundamental_price = float(market_settings["fundamentalPrice"])
            elif "marketPrice" in market_settings:
                fundamental_price = float(market_settings["marketPrice"])
            else:
                raise ValueError(
                    f"fundamentalPrice or marketPrice is required for {name}"
                )
            fundamental_drift: float = 0.0
            if "fundamentalDrift" in market_settings:
                fundamental_price = float(market_settings["fundamentalDrift"])
            fundamental_volatility: float = 0.0
            if "fundamentalVolatility" in market_settings:
                fundamental_volatility = float(market_settings["fundamentalVolatility"])

            for i in range(id_from, id_to + 1):
                market = market_class(
                    market_id=i_market,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if n_markets != 1 else ""),
                )
                i_market += 1
                self.simulator._add_market(market=market, group_name=name)
                self.simulator.fundamentals.add_market(
                    market_id=market.market_id,
                    initial=fundamental_price,
                    drift=fundamental_drift,
                    volatility=fundamental_volatility,
                )
                self._pending_setups.append(
                    (market.setup, {"settings": market_settings})
                )

    def _generate_agents(self, agent_type_names: List[str]) -> None:
        i_agent = 0
        for name in agent_type_names:
            agent_settings: Dict = self.settings[name]
            n_agents = 1
            id_from = 0
            id_to = 0
            if "numAgents" in agent_settings:
                n_agents = int(agent_settings["numAgents"])
                id_to = n_agents - 1
            if "from" in agent_settings or "to" in agent_settings:
                if "from" not in agent_settings or "to" not in agent_settings:
                    raise ValueError(
                        f"both {name}.from and {name}.to are required in json file if you use"
                    )
                if "numAgents" in agent_settings:
                    raise ValueError(
                        f"{name}.numMarkets and ({name}.from or {name}.to) cannot be used at the same time"
                    )
                n_agents = agent_settings["to"] - agent_settings["from"]
                id_from = agent_settings["from"]
                id_to = agent_settings["to"]
            prefix: str
            if "prefix" in agent_settings:
                prefix = agent_settings["prefix"]
            else:
                prefix = name + ("-" if n_agents > 1 else "")
            agent_settings = json_extends(
                whole_json=self.settings,
                parent_name=name,
                target_json=agent_settings,
                excludes_fields=["numAgents", "from", "to", "prefix"],
            )
            if "class" not in agent_settings:
                raise ValueError(f"class is not defined for {name}")
            agent_class: Type[Agent] = find_class(name=agent_settings["class"])
            if not issubclass(agent_class, Agent):
                raise ValueError(
                    f"market class for {name} does not inherit Market class"
                )
            if "markets" not in agent_settings:
                raise ValueError(f"markets is required in {name}")
            accessible_market_names: List[str] = agent_settings["markets"]
            accessible_market_ids: List[int] = sum(
                [
                    list(
                        map(
                            lambda m: m.market_id,
                            self.simulator.markets_group_name2market[x],
                        )
                    )
                    for x in accessible_market_names
                ],
                [],
            )
            for i in range(id_from, id_to + 1):
                agent = agent_class(
                    agent_id=i_agent,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if n_agents != 1 else ""),
                )
                i_agent += 1
                self.simulator._add_agent(agent=agent, group_name=name)
                self._pending_setups.append(
                    (
                        agent.setup,
                        {
                            "settings": agent_settings,
                            "accessible_markets_ids": accessible_market_ids,
                        },
                    )
                )

    def _set_fundamental_correlation(self) -> None:
        if "fundamentalCorrelations" in self.settings["simulation"]:
            corr_settings: Dict = self.settings["simulation"]["fundamentalCorrelations"]
            for key, value in corr_settings.items():
                if key == "pairwise":
                    if (
                        not isinstance(value, list)
                        or sum([len(x) != 3 for x in value]) > 0
                    ):
                        raise ValueError(
                            "simulation.fundamentalCorrelations.pairwise has invalid format data"
                        )
                    for (market1_name, market2_name, corr) in value:
                        market1 = self.simulator.name2market[market1_name]
                        market2 = self.simulator.name2market[market2_name]
                        self.simulator.fundamentals.set_correlation(
                            market_id1=market1.market_id,
                            market_id2=market2.market_id,
                            corr=float(corr),
                        )
                else:
                    raise NotImplementedError(
                        f"{key} for simulation.fundamentalCorrelations is not supported"
                    )

    def _setup(self) -> None:
        if "simulation" not in self.settings:
            raise ValueError("simulation is required in json file")

        if "markets" not in self.settings["simulation"]:
            raise ValueError("simulation.markets is required in json file")
        market_type_names: List[str] = self.settings["simulation"]["markets"]
        if (
            not isinstance(market_type_names, list)
            or sum([not isinstance(m, str) for m in market_type_names]) > 0
        ):
            raise ValueError("simulation.markets in json file have to be list[str]")
        self._generate_markets(market_type_names=market_type_names)
        self._set_fundamental_correlation()

        if "agents" not in self.settings["simulation"]:
            raise ValueError("agents.markets is required in json file")
        agent_type_names: List[str] = self.settings["simulation"]["agents"]
        if (
            not isinstance(agent_type_names, list)
            or sum([not isinstance(m, str) for m in agent_type_names]) > 0
        ):
            raise ValueError("simulation.agents in json file have to be list[str]")
        self._generate_agents(agent_type_names=agent_type_names)

        if "sessions" not in self.settings["simulation"]:
            raise ValueError("simulation.sessions is required in json file")
        session_settings: List[Dict[str, Any]] = self.settings["simulation"]["sessions"]
        if (
            not isinstance(session_settings, list)
            or sum([not isinstance(m, dict) for m in session_settings]) > 0
        ):
            raise ValueError("simulation.sessions in json file have to be List[Dict]")

        # ToDo session processing

        _ = [func(**kwargs) for func, kwargs in self._pending_setups]

    def _run(self) -> None:
        pass
