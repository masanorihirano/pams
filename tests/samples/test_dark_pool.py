import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest import mock

import pytest

from pams.agents import FCNAgent
from pams.logs.base import CancelLog
from pams.logs.base import Logger
from pams.logs.base import MarketStepEndLog
from pams.logs.base import OrderLog
from pams.market import Market
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Cancel
from pams.order import Order
from pams.session import Session
from pams.simulator import Simulator
from samples.dark_pool.dark_pool_fcn_agent import DarkPoolFCNAgent
from samples.dark_pool.dark_pool_market import DarkPoolMarket
from samples.dark_pool.dark_pool_print_logger import DarkPoolPrintLogger
from samples.dark_pool.dark_pool_print_logger import find_market_pair
from samples.dark_pool.dark_pool_print_logger import get_trade_price

MARKET_SETTINGS: Dict[str, Any] = {
    "tickSize": 0.01,
    "marketPrice": 300.0,
    "outstandingShares": 25000,
}


def create_markets(
    logger: Optional[Logger] = None,
) -> Tuple[Simulator, Market, DarkPoolMarket]:
    simulator = Simulator(prng=random.Random(4))
    lit_market = Market(
        market_id=0,
        prng=random.Random(5),
        simulator=simulator,
        name="LitMarket",
        logger=logger,
    )
    dark_pool = DarkPoolMarket(
        market_id=1,
        prng=random.Random(6),
        simulator=simulator,
        name="DarkPoolMarket",
        logger=logger,
    )
    simulator._add_market(market=lit_market)
    simulator._add_market(market=dark_pool)
    lit_market.setup(settings=MARKET_SETTINGS)
    dark_pool.setup(settings={**MARKET_SETTINGS, "markets": ["LitMarket"]})
    lit_market._update_time(next_fundamental_price=300.0)
    dark_pool._update_time(next_fundamental_price=300.0)
    return simulator, lit_market, dark_pool


def add_lit_order(lit_market: Market, is_buy: bool, price: float) -> None:
    lit_market._add_order(
        order=Order(
            agent_id=0,
            market_id=lit_market.market_id,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            volume=1,
            price=price,
        )
    )


def add_dark_order(
    dark_pool: DarkPoolMarket, agent_id: int, is_buy: bool, volume: int
) -> Order:
    order = Order(
        agent_id=agent_id,
        market_id=dark_pool.market_id,
        is_buy=is_buy,
        kind=MARKET_ORDER,
        volume=volume,
    )
    dark_pool._add_order(order=order)
    return order


class TestDarkPoolMarket:
    def test_setup(self) -> None:
        _, lit_market, dark_pool = create_markets()
        assert dark_pool.lit_market is lit_market
        assert dark_pool.tick_size == 0.01
        assert dark_pool.get_market_price() == 300.0

    @pytest.mark.parametrize(
        "markets",
        [None, [], ["LitMarket", "LitMarket"], "LitMarket", ["Unknown"], ["Dark"]],
    )
    def test_setup_invalid_markets(self, markets: Any) -> None:
        simulator, _, _ = create_markets()
        dark_pool = DarkPoolMarket(
            market_id=2, prng=random.Random(7), simulator=simulator, name="Dark"
        )
        simulator._add_market(market=dark_pool)
        settings: Dict[str, Any] = dict(MARKET_SETTINGS)
        if markets is not None:
            settings["markets"] = markets
        with pytest.raises(ValueError):
            dark_pool.setup(settings=settings)

    def test_get_lit_mid_price(self) -> None:
        _, lit_market, dark_pool = create_markets()
        assert dark_pool.get_lit_mid_price() == 300.0
        add_lit_order(lit_market=lit_market, is_buy=True, price=299.0)
        assert dark_pool.get_lit_mid_price() == 300.0
        add_lit_order(lit_market=lit_market, is_buy=False, price=301.01)
        assert dark_pool.get_lit_mid_price() == pytest.approx(300.005)
        lit_market._market_prices[lit_market.get_time()] = 310.0
        assert dark_pool.get_lit_mid_price() == pytest.approx(300.005)

    def test_add_order_rejects_limit_order(self) -> None:
        _, _, dark_pool = create_markets()
        dark_pool._is_running = True
        with pytest.raises(ValueError):
            dark_pool._add_order(
                order=Order(
                    agent_id=0,
                    market_id=dark_pool.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=300.0,
                )
            )
        assert len(dark_pool.buy_order_book) == 0

    def test_add_order_while_not_running(self) -> None:
        logger = Logger()
        _, _, dark_pool = create_markets(logger=logger)
        order = add_dark_order(dark_pool=dark_pool, agent_id=3, is_buy=True, volume=1)
        assert order.is_canceled
        assert len(dark_pool.buy_order_book) == 0
        assert dark_pool.get_n_buy_order() == 1
        order_log, cancel_log = logger.pending_logs
        assert isinstance(order_log, OrderLog)
        assert order_log.order_id == order.order_id
        assert isinstance(cancel_log, CancelLog)
        assert cancel_log.order_id == order.order_id
        order = add_dark_order(dark_pool=dark_pool, agent_id=4, is_buy=False, volume=1)
        assert order.is_canceled
        assert len(dark_pool.sell_order_book) == 0

    def test_add_order_while_running(self) -> None:
        _, _, dark_pool = create_markets()
        dark_pool._is_running = True
        order = add_dark_order(dark_pool=dark_pool, agent_id=3, is_buy=True, volume=1)
        assert not order.is_canceled
        assert dark_pool.buy_order_book.get_best_order() is order

    def test_remain_executable_orders(self) -> None:
        _, _, dark_pool = create_markets()
        dark_pool._is_running = True
        assert not dark_pool.remain_executable_orders()
        add_dark_order(dark_pool=dark_pool, agent_id=1, is_buy=True, volume=2)
        assert not dark_pool.remain_executable_orders()
        add_dark_order(dark_pool=dark_pool, agent_id=2, is_buy=False, volume=1)
        assert dark_pool.remain_executable_orders()

    def test_execution_without_orders(self) -> None:
        _, _, dark_pool = create_markets()
        dark_pool._is_running = True
        assert len(dark_pool._execution()) == 0
        add_dark_order(dark_pool=dark_pool, agent_id=1, is_buy=False, volume=1)
        assert len(dark_pool._execution()) == 0

    @pytest.mark.parametrize(
        "is_buy_later, expected_price", [(True, 300.01), (False, 300.0)]
    )
    def test_execution_at_lit_mid_price(
        self, is_buy_later: bool, expected_price: float
    ) -> None:
        _, lit_market, dark_pool = create_markets()
        add_lit_order(lit_market=lit_market, is_buy=True, price=299.0)
        add_lit_order(lit_market=lit_market, is_buy=False, price=301.01)
        dark_pool._is_running = True
        earlier = add_dark_order(
            dark_pool=dark_pool, agent_id=1, is_buy=not is_buy_later, volume=2
        )
        later = add_dark_order(
            dark_pool=dark_pool, agent_id=2, is_buy=is_buy_later, volume=1
        )
        logs = dark_pool._execution()
        assert len(logs) == 1
        assert logs[0].price == pytest.approx(expected_price)
        assert logs[0].volume == 1
        assert logs[0].buy_agent_id == (2 if is_buy_later else 1)
        assert logs[0].sell_agent_id == (1 if is_buy_later else 2)
        assert earlier.volume == 1
        assert later.volume == 0
        assert (
            len(dark_pool.buy_order_book if is_buy_later else dark_pool.sell_order_book)
            == 0
        )
        assert dark_pool.get_market_price() == pytest.approx(expected_price)
        assert dark_pool.get_executed_volume() == 1
        assert len(lit_market.buy_order_book) == 1
        assert len(lit_market.sell_order_book) == 1
        assert lit_market.get_executed_volume() == 0

    @pytest.mark.parametrize(
        "is_buy_later, expected_price", [(True, 300.01), (False, 300.0)]
    )
    def test_execution_at_lit_market_price(
        self, is_buy_later: bool, expected_price: float
    ) -> None:
        _, lit_market, dark_pool = create_markets()
        add_lit_order(lit_market=lit_market, is_buy=True, price=299.0)
        lit_market._market_prices[lit_market.get_time()] = 300.004
        dark_pool._is_running = True
        add_dark_order(
            dark_pool=dark_pool, agent_id=1, is_buy=not is_buy_later, volume=1
        )
        add_dark_order(dark_pool=dark_pool, agent_id=2, is_buy=is_buy_later, volume=1)
        logs = dark_pool._execution()
        assert [log.price for log in logs] == [pytest.approx(expected_price)]
        assert not dark_pool.remain_executable_orders()
        assert len(dark_pool.buy_order_book) == 0
        assert len(dark_pool.sell_order_book) == 0

    def test_execution_in_time_priority(self) -> None:
        _, _, dark_pool = create_markets()
        dark_pool._is_running = True
        add_dark_order(dark_pool=dark_pool, agent_id=1, is_buy=False, volume=1)
        add_dark_order(dark_pool=dark_pool, agent_id=2, is_buy=False, volume=2)
        add_dark_order(dark_pool=dark_pool, agent_id=3, is_buy=False, volume=1)
        buy_order = add_dark_order(
            dark_pool=dark_pool, agent_id=4, is_buy=True, volume=4
        )
        dark_pool._update_time(next_fundamental_price=300.0)
        add_dark_order(dark_pool=dark_pool, agent_id=5, is_buy=True, volume=1)
        logs = dark_pool._execution()
        assert [(log.sell_agent_id, log.volume) for log in logs] == [
            (1, 1),
            (2, 2),
            (3, 1),
        ]
        assert all(log.buy_agent_id == 4 for log in logs)
        assert all(log.price == 300.0 for log in logs)
        assert buy_order.volume == 0
        assert len(dark_pool.buy_order_book) == 1
        assert len(dark_pool.sell_order_book) == 0
        assert dark_pool.get_executed_volume() == 4


AGENT_SETTINGS: Dict[str, Any] = {
    "assetVolume": 50,
    "cashAmount": 10000,
    "fundamentalWeight": 1.0,
    "chartWeight": 0.0,
    "noiseWeight": 1.0,
    "noiseScale": 0.001,
    "timeWindowSize": 100,
    "orderMargin": 0.1,
}


def create_agent(
    simulator: Simulator,
    dark_pool_chance: Any,
    accessible_markets_ids: Optional[List[int]] = None,
) -> DarkPoolFCNAgent:
    agent = DarkPoolFCNAgent(
        agent_id=0, prng=random.Random(42), simulator=simulator, name="DarkPoolAgent"
    )
    agent.setup(
        settings={**AGENT_SETTINGS, "darkPoolChance": dark_pool_chance},
        accessible_markets_ids=(
            [0, 1] if accessible_markets_ids is None else accessible_markets_ids
        ),
    )
    return agent


def create_fcn_agent(simulator: Simulator) -> FCNAgent:
    agent = FCNAgent(
        agent_id=0, prng=random.Random(42), simulator=simulator, name="FCNAgent"
    )
    agent.setup(settings=AGENT_SETTINGS, accessible_markets_ids=[0, 1])
    return agent


class TestDarkPoolFCNAgent:
    def test_setup(self) -> None:
        simulator, _, _ = create_markets()
        agent = create_agent(simulator=simulator, dark_pool_chance=0.3)
        assert agent.dark_pool_chance == 0.3
        assert agent.time_window_size == 100
        assert agent.is_market_accessible(market_id=0)
        assert agent.is_market_accessible(market_id=1)
        agent = create_agent(simulator=simulator, dark_pool_chance=[0.2, 0.4])
        assert 0.2 <= agent.dark_pool_chance <= 0.4

    @pytest.mark.parametrize("dark_pool_chance", [None, 1.5, -0.1])
    def test_setup_invalid(self, dark_pool_chance: Optional[float]) -> None:
        simulator, _, _ = create_markets()
        agent = DarkPoolFCNAgent(
            agent_id=0, prng=random.Random(42), simulator=simulator, name="Agent"
        )
        settings: Dict[str, Any] = dict(AGENT_SETTINGS)
        if dark_pool_chance is not None:
            settings["darkPoolChance"] = dark_pool_chance
        with pytest.raises(ValueError):
            agent.setup(settings=settings, accessible_markets_ids=[0, 1])

    def test_submit_orders_by_market_to_lit_market(self) -> None:
        simulator, lit_market, _ = create_markets()
        agent = create_agent(simulator=simulator, dark_pool_chance=0.0)
        assert len(agent.submit_orders_by_market(market=lit_market)) == 0

    @pytest.mark.parametrize("accessible_markets_ids", [[0], [1]])
    def test_submit_orders_by_market_inaccessible(
        self, accessible_markets_ids: List[int]
    ) -> None:
        simulator, _, dark_pool = create_markets()
        agent = create_agent(
            simulator=simulator,
            dark_pool_chance=1.0,
            accessible_markets_ids=accessible_markets_ids,
        )
        assert len(agent.submit_orders_by_market(market=dark_pool)) == 0

    def test_submit_orders_by_market_to_dark_pool(self) -> None:
        simulator, lit_market, dark_pool = create_markets()
        agent = create_agent(simulator=simulator, dark_pool_chance=1.0)
        fcn_agent = create_fcn_agent(simulator=simulator)
        lit_orders = fcn_agent.submit_orders_by_market(market=lit_market)
        orders = agent.submit_orders_by_market(market=dark_pool)
        assert len(orders) == len(lit_orders) == 1
        for order, lit_order in zip(orders, lit_orders):
            assert isinstance(order, Order)
            assert isinstance(lit_order, Order)
            assert order.agent_id == agent.agent_id
            assert order.market_id == dark_pool.market_id
            assert order.kind == MARKET_ORDER
            assert order.price is None
            assert order.is_buy == lit_order.is_buy
            assert order.volume == lit_order.volume
            assert order.ttl == lit_order.ttl

    def test_submit_orders_by_market_without_dark_pool_chance(self) -> None:
        simulator, lit_market, dark_pool = create_markets()
        agent = create_agent(simulator=simulator, dark_pool_chance=0.0)
        fcn_agent = create_fcn_agent(simulator=simulator)
        lit_orders = fcn_agent.submit_orders_by_market(market=lit_market)
        orders = agent.submit_orders_by_market(market=dark_pool)
        assert len(orders) == len(lit_orders) == 1
        for order, lit_order in zip(orders, lit_orders):
            assert isinstance(order, Order)
            assert isinstance(lit_order, Order)
            assert order.market_id == lit_market.market_id
            assert order.kind == LIMIT_ORDER
            assert order.price == lit_order.price
            assert order.is_buy == lit_order.is_buy
            assert order.volume == lit_order.volume
            assert order.ttl == lit_order.ttl

    def test_submit_orders_by_market_for_each_order(self) -> None:
        simulator, lit_market, dark_pool = create_markets()
        agent = create_agent(simulator=simulator, dark_pool_chance=0.3)
        buy_order = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=2,
            price=301.0,
            ttl=100,
        )
        sell_order = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=299.0,
            ttl=100,
        )
        cancel = Cancel(order=sell_order)
        with mock.patch.object(
            FCNAgent,
            "submit_orders_by_market",
            return_value=[buy_order, sell_order, cancel],
        ) as submit_mock, mock.patch.object(
            agent.prng, "random", side_effect=[0.2, 0.5]
        ) as random_mock:
            orders = agent.submit_orders_by_market(market=dark_pool)
        submit_mock.assert_called_once_with(market=lit_market)
        assert random_mock.call_count == 2
        dark_order, lit_order, passed_cancel = orders
        assert isinstance(dark_order, Order)
        assert dark_order.market_id == dark_pool.market_id
        assert dark_order.kind == MARKET_ORDER
        assert dark_order.is_buy
        assert dark_order.volume == 2
        assert dark_order.ttl == 100
        assert lit_order is sell_order
        assert passed_cancel is cancel

    def test_submit_orders(self) -> None:
        simulator, lit_market, dark_pool = create_markets()
        agent = create_agent(simulator=simulator, dark_pool_chance=0.5)
        assert len(agent.submit_orders(markets=[lit_market, dark_pool])) == 1
        assert len(agent.submit_orders(markets=[lit_market])) == 0


def execute_lit_orders(lit_market: Market) -> None:
    add_lit_order(lit_market=lit_market, is_buy=True, price=299.0)
    add_lit_order(lit_market=lit_market, is_buy=False, price=301.01)
    add_lit_order(lit_market=lit_market, is_buy=True, price=301.0)
    lit_market._is_running = True
    add_lit_order(lit_market=lit_market, is_buy=False, price=300.5)
    lit_market._execution()


def execute_dark_orders(dark_pool: DarkPoolMarket) -> None:
    dark_pool._is_running = True
    add_dark_order(dark_pool=dark_pool, agent_id=1, is_buy=True, volume=2)
    add_dark_order(dark_pool=dark_pool, agent_id=2, is_buy=False, volume=2)
    dark_pool._execution()


class TestDarkPoolPrintLogger:
    def test_find_market_pair(self) -> None:
        simulator, lit_market, dark_pool = create_markets()
        other_market = Market(
            market_id=2, prng=random.Random(7), simulator=simulator, name="Other"
        )
        markets: List[Market] = [lit_market, dark_pool, other_market]
        assert find_market_pair(market=lit_market, markets=markets) == (
            lit_market,
            dark_pool,
        )
        assert find_market_pair(market=dark_pool, markets=markets) == (
            lit_market,
            dark_pool,
        )
        assert find_market_pair(market=other_market, markets=markets) == (
            other_market,
            None,
        )
        assert find_market_pair(market=lit_market, markets=[lit_market]) == (
            lit_market,
            None,
        )

    def test_get_trade_price(self) -> None:
        _, lit_market, dark_pool = create_markets()
        assert math.isnan(get_trade_price(lit_market=lit_market, dark_pool=dark_pool))
        assert math.isnan(get_trade_price(lit_market=lit_market, dark_pool=None))
        execute_lit_orders(lit_market=lit_market)
        assert lit_market.get_executed_volume() == 1
        assert lit_market.get_market_price() == 301.0
        assert get_trade_price(lit_market=lit_market, dark_pool=dark_pool) == 301.0
        assert get_trade_price(lit_market=lit_market, dark_pool=None) == 301.0
        execute_dark_orders(dark_pool=dark_pool)
        assert dark_pool.get_executed_volume() == 2
        assert get_trade_price(
            lit_market=lit_market, dark_pool=dark_pool
        ) == pytest.approx(300.0)
        assert get_trade_price(lit_market=lit_market, dark_pool=None) == 301.0

    def test_process_market_step_end_log(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        simulator, lit_market, dark_pool = create_markets()
        session = Session(
            session_id=1,
            prng=random.Random(8),
            session_start_time=0,
            simulator=simulator,
            name="1",
        )
        logger = DarkPoolPrintLogger()
        for market in [lit_market, dark_pool]:
            logger.process_market_step_end_log(
                log=MarketStepEndLog(
                    session=session, market=market, simulator=simulator
                )
            )
        execute_lit_orders(lit_market=lit_market)
        for market in [lit_market, dark_pool]:
            logger.process_market_step_end_log(
                log=MarketStepEndLog(
                    session=session, market=market, simulator=simulator
                )
            )
        execute_dark_orders(dark_pool=dark_pool)
        for market in [lit_market, dark_pool]:
            logger.process_market_step_end_log(
                log=MarketStepEndLog(
                    session=session, market=market, simulator=simulator
                )
            )
        assert capsys.readouterr().out.splitlines() == [
            "1 0 0 LitMarket 300.0 300.0 nan 0",
            "1 0 1 DarkPoolMarket 300.0 300.0 nan 0",
            "1 0 0 LitMarket 301.0 300.0 301.0 1",
            "1 0 1 DarkPoolMarket 300.0 300.0 301.0 0",
            "1 0 0 LitMarket 301.0 300.0 300.0 1",
            "1 0 1 DarkPoolMarket 300.0 300.0 300.0 2",
        ]
