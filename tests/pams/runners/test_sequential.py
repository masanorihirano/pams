import copy
import os.path
import random
import time
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from unittest import mock

import pytest
from numpy.linalg import LinAlgError

from pams import LIMIT_ORDER
from pams import Cancel
from pams import Market
from pams import Order
from pams.agents import Agent
from pams.runners import SequentialRunner
from tests.pams.runners.test_base import TestRunner

from .dummy import DummyLogger
from .dummy import DummyLogger2


class TestSequentialRunner(TestRunner):
    runner_class: Type[SequentialRunner] = SequentialRunner
    default_setting: Dict = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["FCNAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 10,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                    "events": ["FundamentalPriceShock"],
                }
            ],
        },
        "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
        "FCNAgents": {
            "class": "FCNAgent",
            "numAgents": 10,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": {"expon": [1.0]},
            "chartWeight": {"expon": [0.0]},
            "noiseWeight": {"expon": [1.0]},
            "meanReversionTime": {"uniform": [50, 100]},
            "noiseScale": 0.001,
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
        },
        "FundamentalPriceShock": {
            "class": "FundamentalPriceShock",
            "target": "Market",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 1,
            "enabled": True,
        },
    }

    def test__(self) -> None:
        config = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "samples",
            "CI2002",
            "config.json",
        )
        runner = self.runner_class(settings=config, prng=random.Random(42))
        runner._setup()
        runner.simulator._update_times_on_markets(markets=runner.simulator.markets)
        start_time = time.time()
        for _ in range(10000):
            _ = runner._collect_orders_from_normal_agents(
                session=runner.simulator.sessions[0]
            )
        end_time = time.time()
        time_per_step = (end_time - start_time) / 1000
        print("time/step", time_per_step)
        assert time_per_step < 0.003

    def test_generate_markets(self) -> None:
        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["market"])
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 1
        market = runner.simulator.markets[0]
        assert market.name == "Market"
        assert len(runner._pending_setups) == 1
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        }
        assert runner.simulator.fundamentals.prices == {0: [300.0]}
        assert runner.simulator.fundamentals.drifts == {0: 0.0}
        assert runner.simulator.fundamentals.volatilities == {0: 0.0}

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "from": 0,
                "to": 10,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Market": {"extends": "MarketBase"},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 1
        market = runner.simulator.markets[0]
        assert market.name == "Market"
        assert len(runner._pending_setups) == 1
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        }

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "numMarkets": 10,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Market": {"extends": "MarketBase"},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 10
        market = runner.simulator.markets[0]
        assert market.name == "Market-0"
        assert list(map(lambda x: x.name, runner.simulator.markets)) == [
            f"Market-{i}" for i in range(10)
        ]
        assert len(runner._pending_setups) == 10
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        }

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 400.0,
                "outstandingShares": 2000,
                "prefix": "Test",
            },
            "Market": {"extends": "MarketBase", "from": 10, "to": 19},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 10
        market = runner.simulator.markets[0]
        assert market.name == "Test10"
        assert list(map(lambda x: x.name, runner.simulator.markets)) == [
            f"Test{i + 10}" for i in range(10)
        ]
        assert len(runner._pending_setups) == 10
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 400.0,
                "outstandingShares": 2000,
            }
        }

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "numMarkets": 10,
                "from": 0,
                "to": 10,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "from": 0,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Agent",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalDrift": 0.1,
                "fundamentalVolatility": 0.2,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        assert runner.simulator.fundamentals.prices == {0: [300.0]}
        assert runner.simulator.fundamentals.drifts == {0: 0.1}
        assert runner.simulator.fundamentals.volatilities == {0: 0.2}

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "outstandingShares": 2000},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

    def test_generate_agents(self) -> None:
        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {"class": "FCNAgent", "markets": ["Market"]},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["agent"])
        runner._generate_agents(agent_type_names=["Agent"])
        assert len(runner.simulator.agents) == 1
        agent = runner.simulator.agents[0]
        assert agent.agent_id == 0
        assert agent.name == "Agent"
        assert agent.simulator == runner.simulator
        assert len(runner._pending_setups) == 2
        assert runner._pending_setups[1][0] == agent.setup
        assert runner._pending_setups[1][1] == {
            "settings": {"class": "FCNAgent", "markets": ["Market"]},
            "accessible_markets_ids": [0],
        }

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "AgentBase": {
                "class": "FCNAgent",
                "numAgents": 10,
                "from": 0,
                "to": 10,
                "prefix": "Test",
                "markets": ["Market"],
            },
            "Agent": {"extends": "AgentBase"},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        runner._generate_agents(agent_type_names=["Agent"])
        assert len(runner.simulator.agents) == 10
        agent = runner.simulator.agents[0]
        assert agent.agent_id == 0
        assert agent.name == "Test0"
        assert len(runner._pending_setups) == 11
        assert runner._pending_setups[1][0] == agent.setup
        assert runner._pending_setups[1][1] == {
            "settings": {"class": "FCNAgent", "markets": ["Market"]},
            "accessible_markets_ids": [0],
        }

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {"markets": ["Market"]},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["Agent"])
        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {"class": "Market", "markets": ["Market"]},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["Agent"])

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {"class": "FCNAgent"},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["Agent"])

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {
                "class": "FCNAgent",
                "numAgents": 10,
                "from": 0,
                "to": 9,
                "markets": ["Market"],
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["Agent"])

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {"class": "FCNAgent", "from": 0, "markets": ["Market"]},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["Agent"])

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Agent": {"class": "FCNAgent", "from": 10, "to": 19, "markets": ["Market"]},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        runner._generate_agents(agent_type_names=["Agent"])
        assert len(runner.simulator.agents) == 10
        agent = runner.simulator.agents[0]
        assert agent.agent_id == 0
        assert agent.name == "Agent-10"
        assert list(map(lambda x: x.name, runner.simulator.agents)) == [
            f"Agent-{10 + i}" for i in range(10)
        ]
        assert len(runner._pending_setups) == 11
        assert runner._pending_setups[1][0] == agent.setup
        assert runner._pending_setups[1][1] == {
            "settings": {"class": "FCNAgent", "markets": ["Market"]},
            "accessible_markets_ids": [0],
        }

    def test_set_fundamental_correlation(self) -> None:
        setting = {
            "simulation": {
                "markets": ["Market"],
                "fundamentalCorrelations": {
                    "pairwise": [
                        ["Market-0", "Market-1", 0.9],
                        ["Market-0", "Market-2", -0.1],
                    ]
                },
            },
            "Market": {
                "class": "Market",
                "numMarkets": 3,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalVolatility": 0.1,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        runner._set_fundamental_correlation()
        assert runner.simulator.fundamentals.correlation == {(0, 1): 0.9, (0, 2): -0.1}

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "numMarkets": 3,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalVolatility": 0.1,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        runner._set_fundamental_correlation()
        assert runner.simulator.fundamentals.correlation == {}

        setting = {
            "simulation": {
                "markets": ["Market"],
                "fundamentalCorrelations": {
                    "unknown": [
                        ["Market-0", "Market-1", 0.9],
                        ["Market-0", "Market-2", -0.1],
                    ]
                },
            },
            "Market": {
                "class": "Market",
                "numMarkets": 3,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalVolatility": 0.1,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(NotImplementedError):
            runner._set_fundamental_correlation()

        setting = {
            "simulation": {
                "markets": ["Market"],
                "fundamentalCorrelations": {
                    "pairwise": [
                        ["Market-0", "Market-1"],
                        ["Market-0", "Market-2", -0.1],
                    ]
                },
            },
            "Market": {
                "class": "Market",
                "numMarkets": 3,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalVolatility": 0.1,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._set_fundamental_correlation()

        setting = {
            "simulation": {
                "markets": ["Market"],
                "fundamentalCorrelations": {
                    "pairwise": [
                        ["Market-0", "Market-1", 0.9],
                        ["Market-0", "Market-2", -0.1],
                    ]
                },
            },
            "Market": {
                "class": "Market",
                "numMarkets": 3,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._set_fundamental_correlation()

        setting = {
            "simulation": {
                "markets": ["Market"],
                "fundamentalCorrelations": {
                    "pairwise": [
                        ["Market-0", "Market-1", 0.9],
                        ["Market-0", "Market-2", -0.1],
                        ["Market-1", "Market-2", 0.5],
                    ]
                },
            },
            "Market": {
                "class": "Market",
                "numMarkets": 3,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalVolatility": 0.1,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        runner._set_fundamental_correlation()
        assert runner.simulator.fundamentals.correlation == {
            (0, 1): 0.9,
            (0, 2): -0.1,
            (1, 2): 0.5,
        }
        with pytest.raises(LinAlgError):
            runner.simulator.fundamentals._generate_next()

    def test_generate_sessions(self) -> None:
        setting = {
            "simulation": {
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 100,
                        "withOrderPlacement": False,
                        "withOrderExecution": False,
                        "withPrint": True,
                        "highFrequencySubmitRate": 0.2,
                        "maxNormalOrders": 1,
                        "maxHighFrequencyOrders": 5,
                    },
                    {
                        "sessionName": 1,
                        "iterationSteps": 500,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": False,
                        "highFrequencySubmitRate": 0.0,
                        "maxNormalOrders": 2,
                        "maxHighFrequencyOrders": 3,
                        "events": ["FundamentalPriceShock"],
                    },
                ]
            },
            "FundamentalPriceShock": {
                "class": "FundamentalPriceShock",
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
                "shockTimeLength": 2,
                "enabled": True,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_sessions()
        assert len(runner.simulator.sessions) == 2
        assert runner.simulator.sessions[0].name == str(0)
        assert runner.simulator.sessions[1].name == str(1)
        assert runner.simulator.sessions[0].session_id == 0
        assert runner.simulator.sessions[1].session_id == 1
        assert runner.simulator.sessions[0].session_start_time == 0
        assert runner.simulator.sessions[1].session_start_time == 100
        assert runner.simulator.sessions[0].simulator == runner.simulator
        assert runner.simulator.sessions[1].simulator == runner.simulator
        assert len(runner._pending_setups) == 4
        assert runner._pending_setups[0][0] == runner.simulator.sessions[0].setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "sessionName": 0,
                "iterationSteps": 100,
                "withOrderPlacement": False,
                "withOrderExecution": False,
                "withPrint": True,
                "highFrequencySubmitRate": 0.2,
                "maxNormalOrders": 1,
                "maxHighFrequencyOrders": 5,
            }
        }
        assert runner._pending_setups[1][0] == runner.simulator.sessions[1].setup
        assert runner._pending_setups[1][1] == {
            "settings": {
                "sessionName": 1,
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": False,
                "highFrequencySubmitRate": 0.0,
                "maxNormalOrders": 2,
                "maxHighFrequencyOrders": 3,
                "events": ["FundamentalPriceShock"],
            }
        }
        assert runner._pending_setups[2][1] == {
            "settings": {
                "class": "FundamentalPriceShock",
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
                "shockTimeLength": 2,
                "enabled": True,
            }
        }

        setting = {
            "simulation": {},
            "FundamentalPriceShock": {
                "class": "FundamentalPriceShock",
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
                "shockTimeLength": 2,
                "enabled": True,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_sessions()

        setting = {
            "simulation": {
                "sessions": {
                    "sessionName": 0,
                    "iterationSteps": 100,
                    "withOrderPlacement": False,
                    "withOrderExecution": False,
                    "withPrint": True,
                    "highFrequencySubmitRate": 0.2,
                    "maxNormalOrders": 1,
                    "maxHighFrequencyOrders": 5,
                }
            },
            "FundamentalPriceShock": {
                "class": "FundamentalPriceShock",
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
                "shockTimeLength": 2,
                "enabled": True,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_sessions()
        setting = {
            "simulation": {
                "sessions": [
                    {
                        "iterationSteps": 100,
                        "withOrderPlacement": False,
                        "withOrderExecution": False,
                        "withPrint": True,
                        "highFrequencySubmitRate": 0.2,
                        "maxNormalOrders": 1,
                        "maxHighFrequencyOrders": 5,
                    }
                ]
            },
            "FundamentalPriceShock": {
                "class": "FundamentalPriceShock",
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
                "shockTimeLength": 2,
                "enabled": True,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_sessions()

        setting = {
            "simulation": {
                "sessions": [
                    {
                        "sessionName": 0,
                        "withOrderPlacement": False,
                        "withOrderExecution": False,
                        "withPrint": True,
                        "highFrequencySubmitRate": 0.2,
                        "maxNormalOrders": 1,
                        "maxHighFrequencyOrders": 5,
                    }
                ]
            }
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_sessions()

        setting = {
            "simulation": {
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 100,
                        "withOrderPlacement": False,
                        "withOrderExecution": False,
                        "withPrint": True,
                        "highFrequencySubmitRate": 0.2,
                        "maxNormalOrders": 1,
                        "maxHighFrequencyOrders": 5,
                    },
                    {
                        "sessionName": 1,
                        "iterationSteps": 500,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": False,
                        "highFrequencySubmitRate": 0.0,
                        "maxNormalOrders": 2,
                        "maxHighFrequencyOrders": 3,
                        "events": ["FundamentalPriceShock"],
                    },
                ]
            },
            "FundamentalPriceShock": {
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
                "shockTimeLength": 2,
                "enabled": True,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._generate_sessions()

    def test_setup(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        runner._setup()
        assert len(runner.simulator.agents) == 10
        assert len(runner.simulator.markets) == 1
        assert len(runner.simulator.sessions) == 1
        assert len(runner.simulator.events) == 1
        assert len(runner.simulator.event_hooks) == 1
        assert runner.simulator.name2market == {"Market": runner.simulator.markets[0]}
        assert runner.simulator.agents_group_name2agent == {
            "FCNAgents": runner.simulator.agents
        }
        assert runner.simulator.events_dict == {
            "order_before": {},
            "order_after": {},
            "cancel_before": {},
            "cancel_after": {},
            "execution_after": {},
            "session_before": {},
            "session_after": {},
            "market_before": {0: runner.simulator.event_hooks},
            "market_after": {},
        }
        assert runner.simulator.n_agents == 10
        assert runner.simulator.n_events == 1
        assert runner.simulator.n_markets == 1
        assert runner.simulator.n_sessions == 1
        assert runner.simulator.normal_frequency_agents == runner.simulator.agents

        setting = copy.deepcopy(self.default_setting)
        del setting["simulation"]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        del setting["simulation"]["markets"]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        del setting["simulation"]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["markets"] = {}
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["markets"] = [10]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        del setting["simulation"]["agents"]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["agents"] = {}
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["agents"] = [10]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        del setting["simulation"]["sessions"]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"] = {}
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"] = [10]
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError):
            runner._setup()

    def test_collect_orders_from_normal_agents(self) -> None:
        setting = {
            "simulation": {
                "markets": ["SpotMarket-1", "SpotMarket-2", "IndexMarket-I"],
                "agents": [
                    "FCNAgents-1",
                    "FCNAgents-2",
                    "FCNAgents-I",
                    "ArbitrageAgents",
                ],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 100,
                        "withOrderPlacement": True,
                        "withOrderExecution": False,
                        "withPrint": True,
                        "maxNormalOrders": 3,
                        "MEMO": "The same number as #markets",
                        "maxHighFrequencyOrders": 0,
                    },
                    {
                        "sessionName": 1,
                        "iterationSteps": 500,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": True,
                        "maxNormalOrders": 3,
                        "MEMO": "The same number as #markets",
                        "maxHighFrequencyOrders": 5,
                        "events": ["FundamentalPriceShock"],
                    },
                ],
            },
            "FundamentalPriceShock": {
                "class": "FundamentalPriceShock",
                "target": "SpotMarket-1",
                "triggerTime": 0,
                "priceChangeRate": -0.3,
                "enabled": True,
            },
            "SpotMarket": {
                "class": "Market",
                "tickSize": 0.00001,
                "marketPrice": 300.0,
                "outstandingShares": 25000,
            },
            "SpotMarket-1": {"extends": "SpotMarket"},
            "SpotMarket-2": {"extends": "SpotMarket"},
            "IndexMarket-I": {
                "class": "IndexMarket",
                "tickSize": 0.00001,
                "marketPrice": 300.0,
                "outstandingShares": 25000,
                "markets": ["SpotMarket-1", "SpotMarket-2"],
            },
            "FCNAgent": {
                "class": "FCNAgent",
                "numAgents": 100,
                "markets": ["Market"],
                "assetVolume": 50,
                "cashAmount": 10000,
                "fundamentalWeight": {"expon": [1.0]},
                "chartWeight": {"expon": [0.0]},
                "noiseWeight": {"expon": [1.0]},
                "noiseScale": 0.001,
                "timeWindowSize": [100, 200],
                "orderMargin": [0.0, 0.1],
            },
            "FCNAgents-1": {"extends": "FCNAgent", "markets": ["SpotMarket-1"]},
            "FCNAgents-2": {"extends": "FCNAgent", "markets": ["SpotMarket-2"]},
            "FCNAgents-I": {"extends": "FCNAgent", "markets": ["IndexMarket-I"]},
            "ArbitrageAgents": {
                "class": "ArbitrageAgent",
                "numAgents": 100,
                "markets": ["IndexMarket-I", "SpotMarket-1", "SpotMarket-2"],
                "assetVolume": 50,
                "cashAmount": 150000,
                "orderVolume": 1,
                "orderThresholdPrice": 1.0,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()

        def dummy_fn(cls: Agent, markets: List[Market]) -> List[Order]:
            return [
                Order(
                    agent_id=cls.agent_id,
                    market_id=markets[0].market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=300.0,
                )
            ]

        with mock.patch("pams.agents.fcn_agent.FCNAgent.submit_orders", dummy_fn):
            results = runner._collect_orders_from_normal_agents(
                session=runner.simulator.sessions[0]
            )
        assert len(results) == 3

        dummy_order = Order(
            agent_id=100,
            market_id=2,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=300.0,
        )

        with mock.patch(
            "pams.agents.fcn_agent.FCNAgent.submit_orders", return_value=[dummy_order]
        ):
            with pytest.raises(ValueError):
                _ = runner._collect_orders_from_normal_agents(
                    session=runner.simulator.sessions[0]
                )

        setting["simulation"]["sessions"][0]["withOrderPlacement"] = False  # type: ignore
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()
        dummy_order = Order(
            agent_id=100,
            market_id=2,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=300.0,
        )

        with mock.patch(
            "pams.agents.fcn_agent.FCNAgent.submit_orders", return_value=[dummy_order]
        ):
            with pytest.raises(AssertionError):
                _ = runner._collect_orders_from_normal_agents(
                    session=runner.simulator.sessions[0]
                )

    def test_handle_orders(self) -> None:
        setting = {
            "simulation": {
                "markets": ["Market"],
                "agents": ["FCNAgents"],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 10,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": True,
                    }
                ],
            },
            "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
            "FCNAgents": {
                "class": "FCNAgent",
                "numAgents": 10,
                "markets": ["Market"],
                "assetVolume": 50,
                "cashAmount": 10000,
                "fundamentalWeight": {"expon": [1.0]},
                "chartWeight": {"expon": [0.0]},
                "noiseWeight": {"expon": [1.0]},
                "meanReversionTime": {"uniform": [50, 100]},
                "noiseScale": 0.001,
                "timeWindowSize": [100, 200],
                "orderMargin": [0.0, 0.1],
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()
        runner.simulator.markets[0]._update_time(next_fundamental_price=200.0)
        local_orders = runner._collect_orders_from_normal_agents(
            session=runner.simulator.sessions[0]
        )
        runner.simulator.sessions[0].with_order_placement = False
        with pytest.raises(AssertionError):
            runner._handle_orders(
                session=runner.simulator.sessions[0], local_orders=local_orders
            )

        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()
        runner.simulator.markets[0]._update_time(next_fundamental_price=200.0)
        dummy_order = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        dummy_cancel = Cancel(order=dummy_order)
        local_orders = [[dummy_order], [dummy_cancel]]
        runner._handle_orders(
            session=runner.simulator.sessions[0], local_orders=local_orders
        )

        class BuggyOrder:
            def __init__(self, market_id: int = 0, agent_id: int = 0):
                self.market_id = market_id
                self.agent_id = agent_id

        local_orders = [[dummy_order], [dummy_cancel], [BuggyOrder()]]  # type: ignore
        with pytest.raises(NotImplementedError):
            runner._handle_orders(
                session=runner.simulator.sessions[0], local_orders=local_orders
            )

        dummy_order1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        dummy_order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        runner.simulator.markets[0]._is_running = True
        local_orders = [[dummy_order1], [dummy_order2]]
        runner._handle_orders(
            session=runner.simulator.sessions[0], local_orders=local_orders
        )

        setting = {
            "simulation": {
                "markets": ["SpotMarket-1", "SpotMarket-2", "IndexMarket-I"],
                "agents": [
                    "FCNAgents-1",
                    "FCNAgents-2",
                    "FCNAgents-I",
                    "ArbitrageAgents",
                ],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 100,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": True,
                        "maxNormalOrders": 3,
                        "MEMO": "The same number as #markets",
                        "maxHighFrequencyOrders": 0,
                        "highFrequencySubmitRate": 0.0,
                    }
                ],
            },
            "SpotMarket": {
                "class": "Market",
                "tickSize": 0.00001,
                "marketPrice": 300.0,
                "outstandingShares": 25000,
            },
            "SpotMarket-1": {"extends": "SpotMarket"},
            "SpotMarket-2": {"extends": "SpotMarket"},
            "IndexMarket-I": {
                "class": "IndexMarket",
                "tickSize": 0.00001,
                "marketPrice": 300.0,
                "outstandingShares": 25000,
                "markets": ["SpotMarket-1", "SpotMarket-2"],
            },
            "FCNAgent": {
                "class": "FCNAgent",
                "numAgents": 100,
                "markets": ["Market"],
                "assetVolume": 50,
                "cashAmount": 10000,
                "fundamentalWeight": {"expon": [1.0]},
                "chartWeight": {"expon": [0.0]},
                "noiseWeight": {"expon": [1.0]},
                "noiseScale": 0.001,
                "timeWindowSize": [100, 200],
                "orderMargin": [0.0, 0.1],
            },
            "FCNAgents-1": {"extends": "FCNAgent", "markets": ["SpotMarket-1"]},
            "FCNAgents-2": {"extends": "FCNAgent", "markets": ["SpotMarket-2"]},
            "FCNAgents-I": {"extends": "FCNAgent", "markets": ["IndexMarket-I"]},
            "ArbitrageAgents": {
                "class": "ArbitrageAgent",
                "numAgents": 100,
                "markets": ["IndexMarket-I", "SpotMarket-1", "SpotMarket-2"],
                "assetVolume": 50,
                "cashAmount": 150000,
                "orderVolume": 1,
                "orderThresholdPrice": 1.0,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()
        runner.simulator.markets[0]._is_running = True
        runner.simulator.markets[1]._is_running = True
        runner.simulator.markets[2]._is_running = True
        runner.simulator.markets[0]._update_time(next_fundamental_price=200.0)
        runner.simulator.markets[1]._update_time(next_fundamental_price=200.0)
        runner.simulator.markets[2]._update_time(next_fundamental_price=200.0)
        local_orders = runner._collect_orders_from_normal_agents(
            session=runner.simulator.sessions[0]
        )
        runner._handle_orders(
            session=runner.simulator.sessions[0], local_orders=local_orders
        )
        dummy_order1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        dummy_order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        local_orders = [[dummy_order1], [dummy_order2]]
        runner.simulator.sessions[0].high_frequency_submission_rate = 1.0
        runner._handle_orders(
            session=runner.simulator.sessions[0], local_orders=local_orders
        )
        runner.simulator.sessions[0].max_high_frequency_orders = 3
        dummy_order1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        dummy_order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        local_orders = [[dummy_order1], [dummy_order2]]

        def dummy_fn(cls: Agent, markets: List[Market]) -> List[Union[Order, Cancel]]:
            d_order = Order(
                agent_id=cls.agent_id,
                market_id=markets[0].market_id,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=300.0,
            )
            d_order2 = Order(
                agent_id=cls.agent_id,
                market_id=markets[0].market_id,
                is_buy=False,
                kind=LIMIT_ORDER,
                volume=1,
                price=300.0,
            )
            d_cancel = Cancel(order=d_order)
            return [d_order, d_order2, d_cancel]

        with mock.patch(
            "pams.agents.arbitrage_agent.ArbitrageAgent.submit_orders", dummy_fn
        ):
            runner._handle_orders(
                session=runner.simulator.sessions[0], local_orders=local_orders
            )

        def dummy_fn2(cls: Agent, markets: List[Market]) -> List[Order]:
            return [
                Order(
                    agent_id=cls.agent_id + 1,
                    market_id=markets[0].market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=300.0,
                )
            ]

        dummy_order1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        dummy_order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        local_orders = [[dummy_order1], [dummy_order2]]
        with mock.patch(
            "pams.agents.arbitrage_agent.ArbitrageAgent.submit_orders", dummy_fn2
        ):
            with pytest.raises(ValueError):
                runner._handle_orders(
                    session=runner.simulator.sessions[0], local_orders=local_orders
                )

        def dummy_fn3(cls: Agent, markets: List[Market]) -> List[Order]:
            d_order = Order(
                agent_id=cls.agent_id,
                market_id=markets[0].market_id,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=300.0,
            )
            d_cancel = Cancel(order=d_order)
            d_bug = BuggyOrder(market_id=markets[0].market_id, agent_id=cls.agent_id)
            return [d_order, d_cancel, d_bug]  # type: ignore

        dummy_order1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        dummy_order2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=100.0,
        )
        local_orders = [[dummy_order1], [dummy_order2]]
        with mock.patch(
            "pams.agents.arbitrage_agent.ArbitrageAgent.submit_orders", dummy_fn3
        ):
            with pytest.raises(NotImplementedError):
                runner._handle_orders(
                    session=runner.simulator.sessions[0], local_orders=local_orders
                )

        runner.simulator.sessions[0].with_order_placement = False
        local_orders = [[], []]
        with mock.patch(
            "pams.agents.arbitrage_agent.ArbitrageAgent.submit_orders", dummy_fn3
        ):
            with pytest.raises(AssertionError):
                runner._handle_orders(
                    session=runner.simulator.sessions[0], local_orders=local_orders
                )

    def test_iterate_market_update(self) -> None:

        logger = DummyLogger()
        runner = self.test__init__(
            setting_mode="dict", logger=logger, simulator_class=None
        )
        runner._setup()
        runner.simulator._update_times_on_markets(runner.simulator.markets)
        runner._iterate_market_updates(runner.simulator.sessions[0])
        assert (
            logger.n_market_step_begin == runner.simulator.sessions[0].iteration_steps
        )
        assert logger.n_market_end_begin == runner.simulator.sessions[0].iteration_steps

    def test_run(self) -> None:
        logger = DummyLogger2()
        runner = self.test__init__(
            setting_mode="dict", logger=logger, simulator_class=None
        )
        runner._setup()
        runner._run()
        assert logger.n_simulation_begin_log == 1
        assert logger.n_simulation_end_log == 1
        assert logger.n_session_begin_log == len(runner.simulator.sessions)
        assert logger.n_session_end_log == len(runner.simulator.sessions)
        assert logger.n_market_step_begin == sum(
            [session.iteration_steps for session in runner.simulator.sessions]
        )
        assert logger.n_market_step_end == sum(
            [session.iteration_steps for session in runner.simulator.sessions]
        )

    def test_collect_orders_from_normal_agents_error_1(self) -> None:
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None
        )
        runner._setup()

        dummy_order = Order(
            agent_id=100,
            market_id=2,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=300.0,
        )

        with mock.patch(
            "pams.agents.fcn_agent.FCNAgent.submit_orders", return_value=[dummy_order]
        ):
            with pytest.raises(ValueError):
                _ = runner._collect_orders_from_normal_agents(
                    session=runner.simulator.sessions[0]
                )

        setting = copy.deepcopy(self.default_setting)
        setting["simulation"]["sessions"][0]["withOrderPlacement"] = False  # type: ignore
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._setup()
        dummy_order = Order(
            agent_id=100,
            market_id=2,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=300.0,
        )

        with mock.patch(
            "pams.agents.fcn_agent.FCNAgent.submit_orders", return_value=[dummy_order]
        ):
            with pytest.raises(AssertionError):
                _ = runner._collect_orders_from_normal_agents(
                    session=runner.simulator.sessions[0]
                )
