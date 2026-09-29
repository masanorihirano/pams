import copy
import json
import os.path
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast
from unittest import mock

import pytest

from pams import Market
from pams import Order
from pams import Simulator
from pams.logs import Logger
from pams.logs import OrderLog
from pams.logs.market_step_loggers import MarketStepSaver
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Cancel
from pams.runners import SequentialRunner
from samples.invest_div.invest_div_fcn_agent import InvestDivFCNAgent
from samples.invest_div.main import main

SAMPLE_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples", "invest_div"
)

BASE_SETTINGS: Dict[str, Any] = {
    "assetVolume": 0,
    "cashAmount": 10000,
    "fundamentalWeight": 1.0,
    "chartWeight": 0.0,
    "noiseWeight": 0.0,
    "noiseScale": 0.001,
    "timeWindowSize": 100,
    "orderMargin": 0.0,
    "leverageRatio": 1.0,
    "diversityRatio": 0.4,
}


def load_sample_config() -> Dict[str, Any]:
    with open(os.path.join(SAMPLE_DIR, "config.json"), encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def create_simulator_and_markets() -> Simulator:
    sim = Simulator(prng=random.Random(4))
    logger = Logger()
    for market_id in range(2):
        market = Market(
            market_id=market_id,
            prng=random.Random(market_id),
            simulator=sim,
            name=f"market{market_id}",
            logger=logger,
        )
        market.setup(
            settings={
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 330.0,
                "outstandingShares": 2000,
            }
        )
        sim._add_market(market=market, group_name="market")
        market._is_running = True
        market._update_time(next_fundamental_price=330.0)
    return sim


def create_agent(
    sim: Simulator, settings: Dict[str, Any], accessible_markets_ids: List[int]
) -> InvestDivFCNAgent:
    agent = InvestDivFCNAgent(
        agent_id=1,
        prng=random.Random(42),
        simulator=sim,
        name="test_agent",
        logger=Logger(),
    )
    agent.setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
    return agent


def add_limit_order(market: Market, is_buy: bool, price: float) -> None:
    market._add_order(
        order=Order(
            agent_id=99,
            market_id=market.market_id,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            volume=1,
            price=price,
            ttl=100,
        )
    )


def assert_fcn_limit_order(order: Order, market_id: int) -> None:
    # the FCN order with the settings above: buy at the expected price 330.
    assert order.market_id == market_id
    assert order.kind == LIMIT_ORDER
    assert order.is_buy
    assert order.price == pytest.approx(330.0)
    assert order.volume == 1
    assert order.ttl == 100


def assert_position_reducing_order(order: Order, market_id: int, is_buy: bool) -> None:
    assert order.market_id == market_id
    assert order.kind == MARKET_ORDER
    assert order.is_buy == is_buy
    assert order.price is None
    assert order.volume == 1
    assert order.ttl == 10


class TestInvestDivFCNAgent:
    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(4))
        agent = create_agent(
            sim=sim,
            settings={**BASE_SETTINGS, "leverageRatio": 2, "diversityRatio": 0.5},
            accessible_markets_ids=[0, 1],
        )
        assert agent.leverage_ratio == 2.0
        assert isinstance(agent.leverage_ratio, float)
        assert agent.diversity_ratio == 0.5
        assert isinstance(agent.diversity_ratio, float)
        assert agent.fundamental_weight == 1.0
        assert agent.get_cash_amount() == 10000.0
        assert agent.get_asset_volume(market_id=1) == 0

    @pytest.mark.parametrize(
        "key, value",
        [
            ("leverageRatio", None),
            ("leverageRatio", "1.0"),
            ("diversityRatio", None),
            ("diversityRatio", "0.4"),
        ],
    )
    def test_setup_error(self, key: str, value: Optional[str]) -> None:
        sim = Simulator(prng=random.Random(4))
        settings = dict(BASE_SETTINGS)
        if value is None:
            del settings[key]
        else:
            settings[key] = value
        with pytest.raises(ValueError):
            create_agent(sim=sim, settings=settings, accessible_markets_ids=[0, 1])

    def test_filter_markets(self) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim, settings=BASE_SETTINGS, accessible_markets_ids=[1]
        )
        assert agent.filter_markets(markets=sim.markets) == [sim.markets[1]]

    def test_get_asset_value(self) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim,
            settings={**BASE_SETTINGS, "assetVolume": 50},
            accessible_markets_ids=[0, 1],
        )
        assert agent.get_asset_value(market=sim.markets[0]) == 15000.0
        agent.set_asset_volume(market_id=1, volume=-10)
        assert agent.get_asset_value(market=sim.markets[1]) == -3000.0

    def test_submit_orders_under_regulation(self) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim,
            settings={**BASE_SETTINGS, "assetVolume": 10},
            accessible_markets_ids=[0, 1],
        )
        # net asset value: 10000 + 3000 * 2 = 16000, asset value in a market: 3000 <= 6400
        orders = cast(List[Order], agent.submit_orders(markets=sim.markets))
        assert len(orders) == 2
        assert_fcn_limit_order(order=orders[0], market_id=0)
        assert_fcn_limit_order(order=orders[1], market_id=1)

    def test_submit_orders_over_leverage_ratio(self) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim, settings=BASE_SETTINGS, accessible_markets_ids=[0, 1]
        )
        agent.set_asset_volume(market_id=0, volume=-20)
        # net asset value: 10000 - 6000 = 4000 < total absolute asset value: 6000
        assert not agent.submit_orders(markets=sim.markets)

    @pytest.mark.parametrize("leverage_ratio, n_orders", [(1.0, 2), (0.99, 0)])
    def test_submit_orders_at_leverage_limit(
        self, leverage_ratio: float, n_orders: int
    ) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim,
            settings={
                **BASE_SETTINGS,
                "assetVolume": 10,
                "cashAmount": 0,
                "leverageRatio": leverage_ratio,
                "diversityRatio": 1.0,
            },
            accessible_markets_ids=[0, 1],
        )
        # net asset value: 6000 = total absolute asset value: 6000
        orders = agent.submit_orders(markets=sim.markets)
        assert len(orders) == n_orders

    @pytest.mark.parametrize(
        "diversity_ratio, is_regulated", [(0.1875, False), (0.18, True)]
    )
    def test_submit_orders_at_diversity_limit(
        self, diversity_ratio: float, is_regulated: bool
    ) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim,
            settings={
                **BASE_SETTINGS,
                "assetVolume": 10,
                "diversityRatio": diversity_ratio,
            },
            accessible_markets_ids=[0, 1],
        )
        # net asset value: 16000, asset value in a market: 3000 = 0.1875 * 16000
        orders = cast(List[Order], agent.submit_orders(markets=sim.markets))
        assert len(orders) == 2
        for market_id, order in enumerate(orders):
            if is_regulated:
                assert_position_reducing_order(
                    order=order, market_id=market_id, is_buy=False
                )
            else:
                assert_fcn_limit_order(order=order, market_id=market_id)

    def test_submit_orders_accessible_markets_only(self) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim,
            settings={**BASE_SETTINGS, "assetVolume": 10},
            accessible_markets_ids=[1],
        )
        # the inaccessible market is neither valued nor ordered.
        # net asset value: 10000 + 3000 = 13000, asset value in market 1: 3000 <= 5200
        orders = cast(List[Order], agent.submit_orders(markets=sim.markets))
        assert len(orders) == 1
        assert_fcn_limit_order(order=orders[0], market_id=1)

    @pytest.mark.parametrize("is_long", [True, False])
    @pytest.mark.parametrize("has_counter_order", [True, False])
    def test_submit_orders_over_diversity_ratio(
        self, is_long: bool, has_counter_order: bool
    ) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim, settings=BASE_SETTINGS, accessible_markets_ids=[0, 1]
        )
        if is_long:
            # net asset value: 10000 + 15000 = 25000, asset value in market 0: 15000 > 10000
            agent.set_asset_volume(market_id=0, volume=50)
        else:
            # net asset value: 30000 - 9000 = 21000, asset value in market 0: -9000, 9000 > 8400
            agent.set_cash_amount(cash_amount=30000)
            agent.set_asset_volume(market_id=0, volume=-30)
        if has_counter_order:
            add_limit_order(market=sim.markets[0], is_buy=is_long, price=300.0)
        # the market order is submitted whether it can be executed immediately or not.
        orders = cast(List[Order], agent.submit_orders(markets=sim.markets))
        assert len(orders) == 2
        assert_position_reducing_order(order=orders[0], market_id=0, is_buy=not is_long)
        assert_fcn_limit_order(order=orders[1], market_id=1)

    def test_submit_orders_without_position(self) -> None:
        sim = create_simulator_and_markets()
        agent = create_agent(
            sim=sim,
            settings={**BASE_SETTINGS, "diversityRatio": -1.0},
            accessible_markets_ids=[0, 1],
        )
        # no position can be reduced although the asset values are over the limit.
        assert not agent.submit_orders(markets=sim.markets)


class MarketOrderRecordingAgent(InvestDivFCNAgent):
    """InvestDivFCNAgent recording its market orders at the submission.

    Each record is (is_buy, volume, ttl, asset volume). They are copied because the volume of an order decreases
    when it is executed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.market_orders: List[Tuple[bool, int, Optional[int], int]] = []

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        orders = super().submit_orders(markets=markets)
        for order in orders:
            if isinstance(order, Order) and order.kind == MARKET_ORDER:
                self.market_orders.append(
                    (
                        order.is_buy,
                        order.volume,
                        order.ttl,
                        self.get_asset_volume(market_id=order.market_id),
                    )
                )
        return orders


class OrderLogSaver(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.order_logs: List[OrderLog] = []

    def process_order_log(self, log: OrderLog) -> None:
        self.order_logs.append(log)


def run_sample(config: Dict[str, Any], seed: int, logger: Logger) -> SequentialRunner:
    runner = SequentialRunner(settings=config, prng=random.Random(seed), logger=logger)
    runner.class_register(cls=InvestDivFCNAgent)
    runner.class_register(cls=MarketOrderRecordingAgent)
    runner.main()
    return runner


def get_market_prices(saver: MarketStepSaver) -> Dict[str, List[float]]:
    prices: Dict[str, List[float]] = {}
    for log in saver.market_step_logs:
        prices.setdefault(log["market_name"], []).append(log["market_price"])
    return prices


def test_market_orders_in_simulation() -> None:
    config = load_sample_config()
    config["InvestDivFCNAgents"]["class"] = "MarketOrderRecordingAgent"
    logger = OrderLogSaver()
    runner = run_sample(config=config, seed=1, logger=logger)

    invest_div_agents = [
        agent
        for agent in runner.simulator.agents
        if isinstance(agent, MarketOrderRecordingAgent)
    ]
    assert len(invest_div_agents) == 100
    market_orders = [
        record for agent in invest_div_agents for record in agent.market_orders
    ]
    # the regulation works in the default setting.
    assert len(market_orders) > 0
    for is_buy, volume, ttl, asset_volume in market_orders:
        assert volume == 1
        assert ttl == 10
        # the market order reduces the position.
        assert asset_volume != 0
        assert is_buy == (asset_volume < 0)

    # all the market orders come from the regulation.
    invest_div_agent_ids = {agent.agent_id for agent in invest_div_agents}
    market_order_logs = [log for log in logger.order_logs if log.kind == MARKET_ORDER]
    assert len(market_order_logs) == len(market_orders)
    for log in market_order_logs:
        assert log.agent_id in invest_div_agent_ids
        assert log.volume == 1


def test_invest_div_fcn_agent_without_binding_regulation() -> None:
    # without the binding regulation, the agent behaves as FCNAgent and uses the same random numbers.
    base_config = load_sample_config()
    base_config["simulation"]["sessions"][1]["iterationSteps"] = 500
    prices: List[Dict[str, List[float]]] = []
    for agent_class in ["InvestDivFCNAgent", "FCNAgent"]:
        config = copy.deepcopy(base_config)
        config["InvestDivFCNAgents"]["class"] = agent_class
        config["InvestDivFCNAgents"]["leverageRatio"] = 1e9
        config["InvestDivFCNAgents"]["diversityRatio"] = 1e9
        saver = MarketStepSaver()
        run_sample(config=config, seed=1, logger=saver)
        prices.append(get_market_prices(saver=saver))
    assert prices[0] == prices[1]
    assert len(prices[0]["Market-1"]) == 600


def test_invest_div() -> None:
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{SAMPLE_DIR}/config.json", "--seed", "1"]
    ):
        main()
