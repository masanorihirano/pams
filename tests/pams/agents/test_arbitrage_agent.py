import random
import re
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

import pytest

from pams import IndexMarket
from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import ArbitrageAgent
from pams.logs import Logger
from pams.runners import SequentialRunner
from tests.pams.agents.test_base import TestAgent

ARBITRAGE_SETTINGS: Dict[str, Any] = {
    "assetVolume": 50,
    "cashAmount": 10000,
    "orderVolume": 1,
    "orderThresholdPrice": 0.1,
}


def add_index_market(sim: Simulator, market_names: List[str], index_name: str) -> None:
    # the component markets and then the index market take the next market IDs
    for name in market_names:
        market = Market(
            market_id=sim.n_markets, prng=random.Random(1), simulator=sim, name=name
        )
        market.setup(
            settings={
                "tickSize": 0.01,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        )
        sim._add_market(market=market, group_name="market")
    index_market = IndexMarket(
        market_id=sim.n_markets, prng=random.Random(1), simulator=sim, name=index_name
    )
    index_market.setup(
        settings={"markets": market_names, "tickSize": 0.01, "fundamentalPrice": 300.0}
    )
    sim._add_market(market=index_market, group_name="index")


def make_index_simulator() -> Simulator:
    # market1 (id 0) and market2 (id 1) compose index (id 2)
    sim = Simulator(prng=random.Random(4))
    add_index_market(sim=sim, market_names=["market1", "market2"], index_name="index")
    return sim


class TestArbitrageAgent(TestAgent):
    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        assert agent.asset_volumes == {0: 50, 1: 50, 2: 50}
        assert agent.cash_amount == 10000
        assert agent.is_market_accessible(0)
        assert agent.is_market_accessible(1)
        assert agent.is_market_accessible(2)
        assert agent.get_asset_volume(0) == 50
        assert agent.get_asset_volume(1) == 50
        assert agent.get_asset_volume(2) == 50
        assert agent.get_cash_amount() == 10000
        assert agent.order_volume == 1
        assert agent.order_threshold_price == 0.1
        assert agent.order_time_length == 10

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings2 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1.1,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings2, accessible_markets_ids=[0, 1, 2])

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings3 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings3, accessible_markets_ids=[0, 1, 2])

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings4 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1.1,
            "orderTimeLength": 10,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings4, accessible_markets_ids=[0, 1, 2])

        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings5 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1,
            "orderThresholdPrice": 0.1,
        }
        agent.setup(settings=settings5, accessible_markets_ids=[0, 1, 2])
        assert agent.order_time_length == 1
        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings6 = {"assetVolume": 50, "cashAmount": 10000, "orderVolume": 1}
        with pytest.raises(ValueError):
            agent.setup(settings=settings6, accessible_markets_ids=[0, 1, 2])
        agent = ArbitrageAgent(
            agent_id=1,
            prng=random.Random(42),
            simulator=sim,
            name="test_agent",
            logger=logger,
        )
        settings5 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 1,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10.1,
        }
        with pytest.raises(ValueError):
            agent.setup(settings=settings5, accessible_markets_ids=[0, 1, 2])

    @pytest.mark.parametrize("index_price", [350.0, 350.05, 349.95, 351, 349])
    @pytest.mark.parametrize("order_volume", [1, 2])
    def test_submit_orders(self, index_price: float, order_volume: int) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        _prng = random.Random(42)
        agent = ArbitrageAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent", logger=logger
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": order_volume,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        market1 = Market(
            market_id=0,
            prng=random.Random(1),
            simulator=sim,
            name="market1",
            logger=logger,
        )
        market1.setup(
            settings={
                "tickSize": 0.01,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        )
        sim._add_market(market=market1, group_name="market")
        market1._is_running = True
        market1._update_time(next_fundamental_price=300.0)
        market2 = Market(
            market_id=1,
            prng=random.Random(1),
            simulator=sim,
            name="market2",
            logger=logger,
        )
        market2.setup(
            settings={
                "tickSize": 0.01,
                "fundamentalPrice": 400.0,
                "outstandingShares": 2000,
            }
        )
        sim._add_market(market=market2, group_name="market")
        market2._is_running = True
        market2._update_time(next_fundamental_price=400.0)
        index_market = IndexMarket(
            market_id=2,
            prng=random.Random(1),
            simulator=sim,
            name="index",
            logger=logger,
        )
        index_market.setup(
            settings={
                "markets": ["market1", "market2"],
                "tickSize": 0.01,
                "fundamentalPrice": index_price,
            }
        )
        index_market._is_running = True
        index_market._update_time(next_fundamental_price=350.0)

        assert index_market.get_index() == 350.0
        assert index_market.get_market_price() == index_price

        orders = agent.submit_orders(markets=[market1, market2, index_market])
        if abs(index_price - 350.0) < 0.1:
            assert len(orders) == 0
        else:
            assert len(orders) == 3
            order1 = cast(Order, [x for x in orders if x.market_id == 0][0])
            order2 = cast(Order, [x for x in orders if x.market_id == 1][0])
            index_order = cast(Order, [x for x in orders if x.market_id == 2][0])
            assert order1.price == 300.0
            assert order2.price == 400.0
            assert index_order.price == index_price
            assert order1.volume == order_volume
            assert order2.volume == order_volume
            assert index_order.volume == 2 * order_volume
            assert order1.ttl == 10
            assert order2.ttl == 10
            assert index_order.ttl == 10
            if index_price > 350.0:
                assert order1.is_buy
                assert order2.is_buy
                assert not index_order.is_buy
            else:
                assert not order1.is_buy
                assert not order2.is_buy
                assert index_order.is_buy
        index_market._is_running = False
        orders = agent.submit_orders(markets=[market1, market2, index_market])
        assert len(orders) == 0
        index_market._is_running = True
        agent2 = ArbitrageAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent", logger=logger
        )
        with pytest.warns(UserWarning, match="cannot access any index market"):
            agent2.setup(settings=settings1, accessible_markets_ids=[0, 1])
        orders = agent2.submit_orders(markets=[market1, market2, index_market])
        assert len(orders) == 0
        market1.outstanding_shares = 1000
        with pytest.raises(NotImplementedError):
            agent.submit_orders(markets=[market1, market2, index_market])

    @pytest.mark.parametrize(
        "accessible_markets_ids, missing_market_names",
        [([2], "market1, market2"), ([0, 2], "market2"), ([2, 1], "market1")],
    )
    def test_setup_warns_inaccessible_components(
        self, accessible_markets_ids: List[int], missing_market_names: str
    ) -> None:
        sim = make_index_simulator()
        agent = ArbitrageAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        with pytest.warns(UserWarning) as record:
            agent.setup(
                settings=ARBITRAGE_SETTINGS,
                accessible_markets_ids=accessible_markets_ids,
            )
        assert [str(w.message) for w in record] == [
            "ArbitrageAgent test_agent can access the index market index but not its component markets "
            f"{missing_market_names}, so its orders to them will fail. "
            "Add the groups of these markets to markets in the settings of this agent."
        ]

    @pytest.mark.parametrize(
        "accessible_markets_ids, expected_messages",
        [
            ([0, 1, 2, 5], ["index2 but not its component markets market3, market4"]),
            (
                [5, 2, 0, 3],
                [
                    "index2 but not its component markets market4",
                    "index but not its component markets market2",
                ],
            ),
            ([0, 1, 2, 3, 4, 5], []),
        ],
    )
    def test_setup_warns_several_index_markets(
        self, accessible_markets_ids: List[int], expected_messages: List[str]
    ) -> None:
        sim = make_index_simulator()
        # market3 (id 3) and market4 (id 4) compose index2 (id 5)
        add_index_market(
            sim=sim, market_names=["market3", "market4"], index_name="index2"
        )
        agent = ArbitrageAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            agent.setup(
                settings=ARBITRAGE_SETTINGS,
                accessible_markets_ids=accessible_markets_ids,
            )
        assert [(w.category, str(w.message)) for w in record] == [
            (
                UserWarning,
                f"ArbitrageAgent test_agent can access the index market {message}, "
                "so its orders to them will fail. "
                "Add the groups of these markets to markets in the settings of this agent.",
            )
            for message in expected_messages
        ]

    @pytest.mark.parametrize("accessible_markets_ids", [[], [0], [0, 1]])
    def test_setup_warns_no_index_market(
        self, accessible_markets_ids: List[int]
    ) -> None:
        sim = make_index_simulator()
        agent = ArbitrageAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        with pytest.warns(UserWarning) as record:
            agent.setup(
                settings=ARBITRAGE_SETTINGS,
                accessible_markets_ids=accessible_markets_ids,
            )
        assert [str(w.message) for w in record] == [
            "ArbitrageAgent test_agent cannot access any index market, so it never submits orders. "
            "Add an index market and its component markets to markets in the settings of this agent."
        ]

    @pytest.mark.parametrize(
        "accessible_markets_ids",
        # 3 and 4 are not registered to the simulator, so they cannot be judged
        [[0, 1, 2], [2, 1, 0], [0, 1, 2, 3], [3], [0, 3], [3, 4]],
    )
    def test_setup_no_warning(self, accessible_markets_ids: List[int]) -> None:
        sim = make_index_simulator()
        agent = ArbitrageAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="test_agent"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent.setup(
                settings=ARBITRAGE_SETTINGS,
                accessible_markets_ids=accessible_markets_ids,
            )

    @pytest.mark.parametrize(
        "markets, expected_warning",
        [
            (["IndexMarket", "SpotMarkets", "SpotMarket"], None),
            (["SpotMarket", "SpotMarkets", "IndexMarket"], None),
            (
                ["IndexMarket"],
                "can access the index market IndexMarket but not its component markets "
                "SpotMarkets-0, SpotMarkets-1, SpotMarket, so",
            ),
            (["IndexMarket", "SpotMarket"], "markets SpotMarkets-0, SpotMarkets-1, so"),
            (["IndexMarket", "SpotMarkets"], "markets SpotMarket, so"),
            (["SpotMarkets", "SpotMarket"], "cannot access any index market"),
        ],
    )
    def test_setup_by_runner(
        self, markets: List[str], expected_warning: Optional[str]
    ) -> None:
        # the runner sets up all the markets, including the index market, before the agents
        setting: Dict[str, Any] = {
            "simulation": {
                "markets": ["SpotMarkets", "SpotMarket", "IndexMarket"],
                "agents": ["ArbitrageAgents"],
                "sessions": [
                    {
                        "sessionName": 0,
                        "iterationSteps": 1,
                        "withOrderPlacement": True,
                        "withOrderExecution": True,
                        "withPrint": False,
                    }
                ],
            },
            "SpotMarkets": {
                "class": "Market",
                "numMarkets": 2,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "SpotMarket": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
            },
            "IndexMarket": {
                "class": "IndexMarket",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "markets": ["SpotMarkets-0", "SpotMarkets-1", "SpotMarket"],
            },
            "ArbitrageAgents": {
                "class": "ArbitrageAgent",
                "numAgents": 2,
                "markets": markets,
                **ARBITRAGE_SETTINGS,
            },
        }
        runner = SequentialRunner(settings=setting, prng=random.Random(42))
        if expected_warning is None:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                runner._setup()
        else:
            with pytest.warns(UserWarning, match=re.escape(expected_warning)) as record:
                runner._setup()
            assert [str(w.message).split(" ")[1] for w in record] == [
                "ArbitrageAgents-0",
                "ArbitrageAgents-1",
            ]
            assert all(expected_warning in str(w.message) for w in record)

    def test__repr__(self) -> None:
        sim = Simulator(prng=random.Random(4))
        logger = Logger()
        _prng = random.Random(42)
        agent = ArbitrageAgent(
            agent_id=1, prng=_prng, simulator=sim, name="test_agent", logger=logger
        )
        settings1 = {
            "assetVolume": 50,
            "cashAmount": 10000,
            "orderVolume": 2,
            "orderThresholdPrice": 0.1,
            "orderTimeLength": 10,
        }
        agent.setup(settings=settings1, accessible_markets_ids=[0, 1, 2])
        assert (
            str(agent)
            == f"<pams.agents.arbitrage_agent.ArbitrageAgent | id=1, rnd={_prng}, order_volume=2, "
            f"order_threshold_price=0.1, order_time_length=10>"
        )
