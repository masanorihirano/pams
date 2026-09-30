import copy
import os.path
import random
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast
from unittest import mock

import pytest
from numpy.linalg import LinAlgError

from pams import LIMIT_ORDER
from pams import Cancel
from pams import Market
from pams import Order
from pams import ProportionalTransactionCost
from pams import TransactionCost
from pams.agents import Agent
from pams.agents import FCNAgent
from pams.events import FundamentalPriceShock
from pams.runners import Runner
from pams.runners import SequentialRunner
from tests.pams.runners.test_base import TestRunner

from .dummy import DummyLogger
from .dummy import DummyLogger2
from .dummy import ExecutionCountLogger
from .dummy import MakerTakerTransactionCost
from .dummy import RandomlyIdleFCNAgent
from .dummy import SimulatorAccessingLogger


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
    TIME_PER_STEP_THRESHOLD: Optional[float] = 0.0003

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
        time_per_step = (end_time - start_time) / 10000
        print("time/step", time_per_step)
        if self.TIME_PER_STEP_THRESHOLD is not None:
            assert time_per_step < self.TIME_PER_STEP_THRESHOLD

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
        assert [x.name for x in runner.simulator.markets] == [
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
        assert [x.name for x in runner.simulator.markets] == [
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

    def _make_from_to_runner(self, setting: Dict) -> SequentialRunner:
        return self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )

    @pytest.mark.parametrize(
        "range_settings, expected_names",
        [
            ({"from": 0, "to": 1}, ["Market-0", "Market-1"]),
            ({"from": 5, "to": 6}, ["Market-5", "Market-6"]),
            ({"from": 0, "to": 1, "prefix": "Test"}, ["Test0", "Test1"]),
            ({"from": 0, "to": 0}, ["Market0"]),
            ({"from": 5, "to": 5}, ["Market5"]),
            ({"from": 5, "to": 5, "prefix": "Test"}, ["Test5"]),
            ({"numMarkets": 1}, ["Market"]),
            ({}, ["Market"]),
        ],
    )
    def test_generate_markets_from_to_naming(
        self, range_settings: Dict, expected_names: List[str]
    ) -> None:
        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                **range_settings,
            },
        }
        runner = self._make_from_to_runner(setting=setting)
        runner._generate_markets(market_type_names=["Market"])
        assert [m.name for m in runner.simulator.markets] == expected_names
        assert [m.market_id for m in runner.simulator.markets] == list(
            range(len(expected_names))
        )
        assert sorted(runner.simulator.name2market.keys()) == sorted(expected_names)
        assert len(runner.simulator.markets_group_name2market["Market"]) == len(
            expected_names
        )
        assert len(runner._pending_setups) == len(expected_names)

    def test_generate_markets_from_to_singletons_shared_prefix(self) -> None:
        setting = {
            "simulation": {"markets": ["MarketA", "MarketB"]},
            "MarketA": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "prefix": "Market",
                "from": 0,
                "to": 0,
            },
            "MarketB": {"extends": "MarketA", "from": 1, "to": 1},
        }
        runner = self._make_from_to_runner(setting=setting)
        runner._generate_markets(market_type_names=["MarketA", "MarketB"])
        assert [m.name for m in runner.simulator.markets] == ["Market0", "Market1"]
        assert len(runner.simulator.name2market) == 2

    def test_generate_markets_reversed_range(self) -> None:
        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "from": 1,
                "to": 0,
            },
        }
        runner = self._make_from_to_runner(setting=setting)
        with pytest.raises(ValueError, match="Market.to"):
            runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 0
        assert len(runner._pending_setups) == 0

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "to": 1,
            },
        }
        runner = self._make_from_to_runner(setting=setting)
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

    def test_generate_markets_with_class(self) -> None:
        class UserDefinedMarket(Market):
            pass

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": UserDefinedMarket,
                "tickSize": 0.01,
                "marketPrice": 300.0,
            },
            "Market": {"extends": "MarketBase", "numMarkets": 2},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        # the class is used without registration
        runner._generate_markets(market_type_names=["Market"])
        assert not runner.registered_classes
        assert [type(market) for market in runner.simulator.markets] == [
            UserDefinedMarket,
            UserDefinedMarket,
        ]
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": UserDefinedMarket,
                "tickSize": 0.01,
                "marketPrice": 300.0,
            }
        }

    @pytest.mark.parametrize(
        "market_class, match",
        [
            (Agent, "market class for Market does not inherit Market class"),
            ("FCNAgent", "market class for Market does not inherit Market class"),
            ("LIMIT_ORDER", "market class for Market does not inherit Market class"),
            (
                None,
                r"^class for Market must be a class name \(str\) or a class, "
                r"but None is given$",
            ),
            (1, "class for Market must be a class name"),
        ],
    )
    def test_generate_markets_with_invalid_class(
        self, market_class: Any, match: str
    ) -> None:
        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {"class": market_class, "tickSize": 0.01, "marketPrice": 300.0},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError, match=match):
            runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 0

    @pytest.mark.parametrize("has_key", [False, True])
    def test_generate_markets_without_transaction_cost(self, has_key: bool) -> None:
        setting: Dict[str, Any] = {
            "simulation": {"markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
        }
        if has_key:
            setting["Market"]["transactionCost"] = None
        runner = self.test__init__(
            setting_mode="dict",
            logger=None,
            simulator_class=None,
            setting=copy.deepcopy(setting),
        )
        runner._generate_markets(market_type_names=["Market"])
        market = runner.simulator.markets[0]
        assert market.transaction_cost is None
        assert runner._pending_setups == [
            (market.setup, {"settings": setting["Market"]})
        ]

    @pytest.mark.parametrize(
        "cost_class, cost_settings, registered, expected_class",
        [
            (
                "ProportionalTransactionCost",
                {"rate": 0.001},
                False,
                ProportionalTransactionCost,
            ),
            (
                ProportionalTransactionCost,
                {"rate": 0.001},
                False,
                ProportionalTransactionCost,
            ),
            (
                "MakerTakerTransactionCost",
                {"makerRate": -0.0001, "takerRate": 0.0003},
                True,
                MakerTakerTransactionCost,
            ),
            (
                MakerTakerTransactionCost,
                {"makerRate": -0.0001, "takerRate": 0.0003},
                False,
                MakerTakerTransactionCost,
            ),
        ],
    )
    def test_generate_markets_with_transaction_cost(
        self,
        cost_class: Union[str, Type],
        cost_settings: Dict[str, Any],
        registered: bool,
        expected_class: Type[TransactionCost],
    ) -> None:
        setting: Dict[str, Any] = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "transactionCost": {"class": cost_class, **cost_settings},
            },
        }
        runner = self.test__init__(
            setting_mode="dict",
            logger=None,
            simulator_class=None,
            setting=copy.deepcopy(setting),
        )
        if registered:
            runner.class_register(cls=expected_class)
        runner._generate_markets(market_type_names=["Market"])
        market = runner.simulator.markets[0]
        transaction_cost = market.transaction_cost
        assert type(transaction_cost) is expected_class
        assert isinstance(transaction_cost, TransactionCost)
        assert transaction_cost.market is market
        # "transactionCost" is left in the settings of the market, and the settings of
        # the transaction cost do not include "class"
        assert runner._pending_setups == [
            (market.setup, {"settings": setting["Market"]}),
            (transaction_cost.setup, {"settings": cost_settings}),
        ]
        for func, kwargs in runner._pending_setups:
            func(**kwargs)
        if isinstance(transaction_cost, ProportionalTransactionCost):
            assert transaction_cost.rate == 0.001
        else:
            assert isinstance(transaction_cost, MakerTakerTransactionCost)
            assert transaction_cost.maker_rate == -0.0001
            assert transaction_cost.taker_rate == 0.0003

    def test_generate_markets_transaction_cost_per_market(self) -> None:
        setup_calls: List[Tuple[TransactionCost, Dict[str, Any], float]] = []

        class RecordingTransactionCost(ProportionalTransactionCost):
            def setup(
                self, settings: Dict[str, Any], *args: Any, **kwargs: Any
            ) -> None:
                setup_calls.append(
                    (self, copy.deepcopy(settings), self.market.tick_size)
                )
                super().setup(settings, *args, **kwargs)

        cost_setting = {"class": RecordingTransactionCost, "rate": 0.001}
        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "transactionCost": cost_setting,
            },
            "Market": {"extends": "MarketBase", "numMarkets": 2},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        markets = runner.simulator.markets
        assert len(markets) == 2
        costs = [market.transaction_cost for market in markets]
        assert all(type(cost) is RecordingTransactionCost for cost in costs)
        assert costs[0] is not costs[1]
        assert [cast(TransactionCost, cost).market for cost in costs] == markets
        assert [func for func, _ in runner._pending_setups] == [
            markets[0].setup,
            cast(TransactionCost, costs[0]).setup,
            markets[1].setup,
            cast(TransactionCost, costs[1]).setup,
        ]
        for func, kwargs in runner._pending_setups:
            func(**kwargs)
        # each transaction cost is set up once, after its market, without "class"
        assert setup_calls == [
            (costs[0], {"rate": 0.001}, 0.01),
            (costs[1], {"rate": 0.001}, 0.01),
        ]
        assert [cast(ProportionalTransactionCost, cost).rate for cost in costs] == [
            0.001,
            0.001,
        ]
        assert cost_setting == {"class": RecordingTransactionCost, "rate": 0.001}

    @pytest.mark.parametrize(
        "transaction_cost, error, match",
        [
            (
                0.001,
                ValueError,
                r"^Market\.transactionCost must be an object or null, "
                r"but 0\.001 is given$",
            ),
            (
                True,
                ValueError,
                r"^Market\.transactionCost must be an object or null, "
                r"but True is given$",
            ),
            (
                "ProportionalTransactionCost",
                ValueError,
                r"^Market\.transactionCost must be an object or null, "
                r"but 'ProportionalTransactionCost' is given$",
            ),
            (
                [{"class": "ProportionalTransactionCost", "rate": 0.001}],
                ValueError,
                r"^Market\.transactionCost must be an object or null, but \[",
            ),
            (
                {"rate": 0.001},
                ValueError,
                r"^class is not defined for Market\.transactionCost$",
            ),
            (
                {"class": None, "rate": 0.001},
                ValueError,
                r"^class for Market\.transactionCost must be a class name \(str\) "
                r"or a class, but None is given$",
            ),
            (
                {"class": "UnknownTransactionCost", "rate": 0.001},
                AttributeError,
                r"^class for UnknownTransactionCost is found 0 times$",
            ),
            (
                {"class": "MakerTakerTransactionCost"},
                AttributeError,
                r"^class for MakerTakerTransactionCost is found 0 times$",
            ),
            (
                {"class": "Market"},
                ValueError,
                r"^transaction cost class for Market does not inherit "
                r"TransactionCost class$",
            ),
            (
                {"class": FCNAgent},
                ValueError,
                r"^transaction cost class for Market does not inherit "
                r"TransactionCost class$",
            ),
            (
                {"class": "LIMIT_ORDER"},
                ValueError,
                r"^transaction cost class for Market does not inherit "
                r"TransactionCost class$",
            ),
        ],
    )
    def test_generate_markets_with_invalid_transaction_cost(
        self, transaction_cost: Any, error: Type[Exception], match: str
    ) -> None:
        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "numMarkets": 2,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "transactionCost": transaction_cost,
            },
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(error, match=match):
            runner._generate_markets(market_type_names=["Market"])
        # the setting is checked before any market is created
        assert len(runner.simulator.markets) == 0
        assert not runner._pending_setups

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
        with pytest.raises(
            ValueError,
            match=r"Agent\.numAgents and \(Agent\.from or Agent\.to\) cannot be used",
        ):
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
        assert [x.name for x in runner.simulator.agents] == [
            f"Agent-{10 + i}" for i in range(10)
        ]
        assert len(runner._pending_setups) == 11
        assert runner._pending_setups[1][0] == agent.setup
        assert runner._pending_setups[1][1] == {
            "settings": {"class": "FCNAgent", "markets": ["Market"]},
            "accessible_markets_ids": [0],
        }

    @pytest.mark.parametrize(
        "range_settings, expected_names",
        [
            ({"from": 0, "to": 1}, ["Agent-0", "Agent-1"]),
            ({"from": 5, "to": 6}, ["Agent-5", "Agent-6"]),
            ({"from": 0, "to": 1, "prefix": "Test"}, ["Test0", "Test1"]),
            ({"from": 0, "to": 0}, ["Agent0"]),
            ({"from": 5, "to": 5}, ["Agent5"]),
            ({"from": 5, "to": 5, "prefix": "Test"}, ["Test5"]),
            ({"numAgents": 1}, ["Agent"]),
            ({}, ["Agent"]),
        ],
    )
    def test_generate_agents_from_to_naming(
        self, range_settings: Dict, expected_names: List[str]
    ) -> None:
        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "Agent": {"class": "FCNAgent", "markets": ["Market"], **range_settings},
        }
        runner = self._make_from_to_runner(setting=setting)
        runner._generate_markets(market_type_names=["Market"])
        runner._generate_agents(agent_type_names=["Agent"])
        assert [a.name for a in runner.simulator.agents] == expected_names
        assert [a.agent_id for a in runner.simulator.agents] == list(
            range(len(expected_names))
        )
        assert sorted(runner.simulator.name2agent.keys()) == sorted(expected_names)
        assert len(runner.simulator.agents_group_name2agent["Agent"]) == len(
            expected_names
        )
        assert len(runner._pending_setups) == 1 + len(expected_names)

    def test_generate_agents_from_to_singletons_shared_prefix(self) -> None:
        setting = {
            "simulation": {"agents": ["AgentA", "AgentB"], "markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "AgentA": {
                "class": "FCNAgent",
                "markets": ["Market"],
                "prefix": "Agent",
                "from": 0,
                "to": 0,
            },
            "AgentB": {"extends": "AgentA", "from": 1, "to": 1},
        }
        runner = self._make_from_to_runner(setting=setting)
        runner._generate_markets(market_type_names=["Market"])
        runner._generate_agents(agent_type_names=["AgentA", "AgentB"])
        assert [a.name for a in runner.simulator.agents] == ["Agent0", "Agent1"]
        assert len(runner.simulator.name2agent) == 2

    def test_generate_agents_reversed_range(self) -> None:
        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "Agent": {"class": "FCNAgent", "from": 1, "to": 0, "markets": ["Market"]},
        }
        runner = self._make_from_to_runner(setting=setting)
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError, match="Agent.to"):
            runner._generate_agents(agent_type_names=["Agent"])
        assert len(runner.simulator.agents) == 0
        assert len(runner._pending_setups) == 1

        setting = {
            "simulation": {"agents": ["Agent"], "markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "Agent": {"class": "FCNAgent", "to": 1, "markets": ["Market"]},
        }
        runner = self._make_from_to_runner(setting=setting)
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError):
            runner._generate_agents(agent_type_names=["Agent"])

    def test_generate_agents_with_class(self) -> None:
        class UserDefinedAgent(FCNAgent):
            pass

        setting = {
            "simulation": {"agents": ["Agents"], "markets": ["Market"]},
            "Market": {"class": Market, "tickSize": 0.01, "marketPrice": 300.0},
            "AgentBase": {"class": UserDefinedAgent, "markets": ["Market"]},
            "Agents": {"extends": "AgentBase", "numAgents": 3},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        # the classes are used without registration
        runner._generate_markets(market_type_names=["Market"])
        runner._generate_agents(agent_type_names=["Agents"])
        assert not runner.registered_classes
        assert [type(market) for market in runner.simulator.markets] == [Market]
        assert [type(agent) for agent in runner.simulator.agents] == [
            UserDefinedAgent,
            UserDefinedAgent,
            UserDefinedAgent,
        ]
        assert runner._pending_setups[1][1] == {
            "settings": {"class": UserDefinedAgent, "markets": ["Market"]},
            "accessible_markets_ids": [0],
        }

    @pytest.mark.parametrize(
        "agent_class, match",
        [
            (Market, "agent class for Agents does not inherit Agent class"),
            ("Market", "agent class for Agents does not inherit Agent class"),
            ("LIMIT_ORDER", "agent class for Agents does not inherit Agent class"),
            (None, "class for Agents must be a class name"),
            (1, "class for Agents must be a class name"),
        ],
    )
    def test_generate_agents_with_invalid_class(
        self, agent_class: Any, match: str
    ) -> None:
        setting = {
            "simulation": {"agents": ["Agents"], "markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "marketPrice": 300.0},
            "Agents": {"class": agent_class, "markets": ["Market"]},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        runner._generate_markets(market_type_names=["Market"])
        with pytest.raises(ValueError, match=match):
            runner._generate_agents(agent_type_names=["Agents"])
        assert len(runner.simulator.agents) == 0

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

    def test_generate_sessions_with_class(self) -> None:
        class UserDefinedEvent(FundamentalPriceShock):
            pass

        setting = {
            "simulation": {
                "sessions": [
                    {"sessionName": 0, "iterationSteps": 100, "events": ["Shock"]}
                ]
            },
            "ShockBase": {
                "class": UserDefinedEvent,
                "target": "Market",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
            },
            "Shock": {"extends": "ShockBase"},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        # the class is used without registration
        runner._generate_sessions()
        assert not runner.registered_classes
        assert len(runner._pending_setups) == 3
        assert runner._pending_setups[1][1] == {
            "settings": {
                "class": UserDefinedEvent,
                "target": "Market",
                "triggerTime": 0,
                "priceChangeRate": -0.1,
            }
        }
        event = runner._pending_setups[2][1]["_event"]
        assert isinstance(event, UserDefinedEvent)
        assert event.name == "Shock"

    @pytest.mark.parametrize(
        "event_class, match",
        [
            (Market, "event class for Shock does not inherit EventABC class"),
            ("FCNAgent", "event class for Shock does not inherit EventABC class"),
            ("LIMIT_ORDER", "event class for Shock does not inherit EventABC class"),
            (None, "class for Shock must be a class name"),
            (1, "class for Shock must be a class name"),
        ],
    )
    def test_generate_sessions_with_invalid_class(
        self, event_class: Any, match: str
    ) -> None:
        setting = {
            "simulation": {
                "sessions": [
                    {"sessionName": 0, "iterationSteps": 100, "events": ["Shock"]}
                ]
            },
            "Shock": {"class": event_class},
        }
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=setting
        )
        with pytest.raises(ValueError, match=match):
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
        with pytest.raises(
            ValueError, match=r"simulation\.agents is required in json file"
        ):
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
            session.iteration_steps for session in runner.simulator.sessions
        )
        assert logger.n_market_step_end == sum(
            session.iteration_steps for session in runner.simulator.sessions
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

    def test_run_execution_logs_processed_once(self) -> None:
        logger = ExecutionCountLogger()
        runner = self.test__init__(
            setting_mode="dict", logger=logger, simulator_class=None
        )
        runner._setup()
        runner._run()
        assert len(logger.execution_logs) > 0
        assert len(logger.execution_logs) == len(
            {id(log) for log in logger.execution_logs}
        )
        assert sum(log.volume for log in logger.execution_logs) == sum(
            sum(market._executed_volumes) for market in runner.simulator.markets
        )

    def test_run_transaction_costs(self) -> None:
        def run(
            has_key: bool, transaction_cost: Optional[Dict[str, Any]] = None
        ) -> Tuple[Runner, ExecutionCountLogger]:
            setting = copy.deepcopy(self.default_setting)
            if has_key:
                setting["Market"]["transactionCost"] = transaction_cost
            logger = ExecutionCountLogger()
            runner = self.test__init__(
                setting_mode="dict",
                logger=logger,
                simulator_class=None,
                setting=setting,
            )
            runner._setup()
            runner._run()
            return runner, logger

        rate = 0.001
        base_runner, base_logger = run(has_key=False)
        null_runner, null_logger = run(has_key=True, transaction_cost=None)
        zero_runner, zero_logger = run(
            has_key=True,
            transaction_cost={"class": "ProportionalTransactionCost", "rate": 0.0},
        )
        cost_runner, cost_logger = run(
            has_key=True,
            transaction_cost={"class": "ProportionalTransactionCost", "rate": rate},
        )
        assert len(base_logger.execution_logs) > 0
        for runner in [base_runner, null_runner]:
            for market in runner.simulator.markets:
                assert market.transaction_cost is None
        for runner, expected_rate in [(zero_runner, 0.0), (cost_runner, rate)]:
            for market in runner.simulator.markets:
                assert isinstance(market.transaction_cost, ProportionalTransactionCost)
                assert market.transaction_cost.rate == expected_rate

        # The runner and the transaction costs draw no random numbers, and the built-in
        # agents ignore their cash, so the costs do not change the prices, the
        # executions and the asset volumes.
        for runner in [null_runner, zero_runner, cost_runner]:
            for base_market, market in zip(
                base_runner.simulator.markets, runner.simulator.markets
            ):
                assert market.get_market_prices() == base_market.get_market_prices()
                assert (
                    market.get_executed_volumes() == base_market.get_executed_volumes()
                )
            for base_agent, agent in zip(
                base_runner.simulator.agents, runner.simulator.agents
            ):
                assert agent.asset_volumes == base_agent.asset_volumes
        for logs in [
            null_logger.execution_logs,
            zero_logger.execution_logs,
            cost_logger.execution_logs,
        ]:
            assert [
                (log.time, log.buy_agent_id, log.sell_agent_id, log.price, log.volume)
                for log in logs
            ] == [
                (log.time, log.buy_agent_id, log.sell_agent_id, log.price, log.volume)
                for log in base_logger.execution_logs
            ]

        # Without costs, the cash is exactly the same.
        for log in (
            base_logger.execution_logs
            + null_logger.execution_logs
            + zero_logger.execution_logs
        ):
            assert log.buy_transaction_cost == 0.0
            assert log.sell_transaction_cost == 0.0
        for runner in [null_runner, zero_runner]:
            assert [agent.cash_amount for agent in runner.simulator.agents] == [
                agent.cash_amount for agent in base_runner.simulator.agents
            ]

        # With costs, both sides pay the rate times the executed value.
        paid: Dict[int, float] = {}
        for log in cost_logger.execution_logs:
            expected = rate * log.price * log.volume
            assert log.buy_transaction_cost == expected
            assert log.sell_transaction_cost == expected
            paid[log.buy_agent_id] = (
                paid.get(log.buy_agent_id, 0.0) + log.buy_transaction_cost
            )
            paid[log.sell_agent_id] = (
                paid.get(log.sell_agent_id, 0.0) + log.sell_transaction_cost
            )
        assert sum(paid.values()) > 0.0
        for base_agent, agent in zip(
            base_runner.simulator.agents, cost_runner.simulator.agents
        ):
            assert agent.cash_amount == pytest.approx(
                base_agent.cash_amount - paid.get(agent.agent_id, 0.0)
            )

    def _check_run_user_transaction_cost(
        self,
        setting: Dict[str, Any],
        by_name: bool,
        runner_class: Optional[Type[SequentialRunner]] = None,
    ) -> List[Tuple[int, int, float, int, float, float]]:
        """Run with and without MakerTakerTransactionCost and check the costs.

        Returns:
            List[Tuple[int, int, float, int, float, float]]: the order IDs, the price, the
            volume and the costs of the executions with the costs.
        """
        maker_rate = -0.0001
        taker_rate = 0.0003
        results: List[Tuple[Runner, ExecutionCountLogger]] = []
        for transaction_cost in [
            None,
            {
                "class": "MakerTakerTransactionCost"
                if by_name
                else MakerTakerTransactionCost,
                "makerRate": maker_rate,
                "takerRate": taker_rate,
            },
        ]:
            run_setting = copy.deepcopy(setting)
            run_setting["Market"]["transactionCost"] = transaction_cost
            logger = ExecutionCountLogger()
            runner = (runner_class or self.runner_class)(
                settings=run_setting, prng=random.Random(42), logger=logger
            )
            if by_name:
                runner.class_register(cls=MakerTakerTransactionCost)
            runner._setup()
            runner._run()
            results.append((runner, logger))
        (base_runner, base_logger), (cost_runner, cost_logger) = results
        for market in cost_runner.simulator.markets:
            assert isinstance(market.transaction_cost, MakerTakerTransactionCost)
        assert len(base_logger.execution_logs) > 0
        assert [
            (log.time, log.buy_order_id, log.sell_order_id, log.price, log.volume)
            for log in cost_logger.execution_logs
        ] == [
            (log.time, log.buy_order_id, log.sell_order_id, log.price, log.volume)
            for log in base_logger.execution_logs
        ]
        paid: Dict[int, float] = {}
        for log in cost_logger.execution_logs:
            value = log.price * log.volume
            # order IDs increase with the arrival of orders in each market, so the
            # order with the smaller ID was placed first and is the maker
            if log.buy_order_id < log.sell_order_id:
                expected = (maker_rate * value, taker_rate * value)
            else:
                expected = (taker_rate * value, maker_rate * value)
            assert (log.buy_transaction_cost, log.sell_transaction_cost) == expected
            paid[log.buy_agent_id] = (
                paid.get(log.buy_agent_id, 0.0) + log.buy_transaction_cost
            )
            paid[log.sell_agent_id] = (
                paid.get(log.sell_agent_id, 0.0) + log.sell_transaction_cost
            )
        for base_agent, agent in zip(
            base_runner.simulator.agents, cost_runner.simulator.agents
        ):
            assert agent.asset_volumes == base_agent.asset_volumes
            assert agent.cash_amount == pytest.approx(
                base_agent.cash_amount - paid.get(agent.agent_id, 0.0)
            )
        return [
            (
                log.buy_order_id,
                log.sell_order_id,
                log.price,
                log.volume,
                log.buy_transaction_cost,
                log.sell_transaction_cost,
            )
            for log in cost_logger.execution_logs
        ]

    @pytest.mark.parametrize("by_name", [True, False])
    def test_run_user_transaction_cost(self, by_name: bool) -> None:
        self._check_run_user_transaction_cost(
            setting=copy.deepcopy(self.default_setting), by_name=by_name
        )

    def test_run_logger_can_access_simulator(self) -> None:
        logger = SimulatorAccessingLogger()
        runner = self.test__init__(
            setting_mode="dict", logger=logger, simulator_class=None
        )
        runner._setup()
        runner._run()
        assert logger.accessed_simulators == [runner.simulator, runner.simulator]

    def test_run_with_class(self) -> None:
        # the runner with classes in the settings gives the same result as the runner
        # with class names and registered classes
        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "RandomlyIdleFCNAgent"
        name_runner = self.runner_class(
            settings=copy.deepcopy(setting),
            prng=random.Random(42),
            logger=DummyLogger2(),
        )
        name_runner.class_register(cls=RandomlyIdleFCNAgent)
        setting["Market"]["class"] = Market
        setting["FCNAgents"]["class"] = RandomlyIdleFCNAgent
        setting["FundamentalPriceShock"]["class"] = FundamentalPriceShock
        class_runner = self.runner_class(
            settings=setting, prng=random.Random(42), logger=DummyLogger2()
        )
        for runner in [name_runner, class_runner]:
            runner._setup()
            runner._run()
        assert not class_runner.registered_classes
        assert [type(agent) for agent in class_runner.simulator.agents] == [
            RandomlyIdleFCNAgent
        ] * len(class_runner.simulator.agents)
        name_market = name_runner.simulator.markets[0]
        class_market = class_runner.simulator.markets[0]
        times = range(name_market.get_time() + 1)
        assert name_market.get_market_prices(times) == class_market.get_market_prices(
            times
        )
        assert name_market.get_fundamental_prices(
            times
        ) == class_market.get_fundamental_prices(times)
        assert [
            (agent.cash_amount, dict(agent.asset_volumes))
            for agent in name_runner.simulator.agents
        ] == [
            (agent.cash_amount, dict(agent.asset_volumes))
            for agent in class_runner.simulator.agents
        ]
        name_logger = name_runner.logger
        class_logger = class_runner.logger
        assert isinstance(name_logger, DummyLogger2)
        assert isinstance(class_logger, DummyLogger2)
        assert class_logger.n_order_log > 0
        assert name_logger.n_order_log == class_logger.n_order_log
        assert name_logger.n_execution_log == class_logger.n_execution_log
