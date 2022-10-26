import random
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union

from .. import Market
from ..utils.class_finder import find_class
from ..utils.json_extends import json_extends
from .base import Runner


class SequentialRunner(Runner):
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
            for i in range(id_from, id_to + 1):
                market = market_class(
                    market_id=i_market,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if n_markets == 1 else ""),
                )
                i_market += 1
                self.simulator._add_market(market=market)

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

        if "agents" not in self.settings["simulation"]:
            raise ValueError("agents.markets is required in json file")
        agent_type_names: List[str] = self.settings["simulation"]["agents"]
        if (
            not isinstance(agent_type_names, list)
            or sum([not isinstance(m, str) for m in agent_type_names]) > 0
        ):
            raise ValueError("simulation.agents in json file have to be list[str]")

        if "sessions" not in self.settings["simulation"]:
            raise ValueError("simulation.sessions is required in json file")
        session_settings: List[Dict[str, Any]] = self.settings["simulation"]["sessions"]
        if (
            not isinstance(session_settings, list)
            or sum([not isinstance(m, dict) for m in session_settings]) > 0
        ):
            raise ValueError("simulation.sessions in json file have to be List[Dict]")

        # ToDo
        fundamental_correlations_settings: Dict[str, List[List[Union[str, float]]]] = {}
        if "fundamentalCorrelations" in self.settings["simulation"]:
            fundamental_correlations_settings = self.settings["simulation"][
                "fundamentalCorrelations"
            ]

    def _run(self) -> None:
        pass
