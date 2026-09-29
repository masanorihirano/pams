import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import pytest

from pams.logs.base import CancelLog
from pams.logs.base import Logger
from pams.logs.base import OrderLog
from pams.market import Market
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from pams.simulator import Simulator
from samples.dark_pool.dark_pool_market import DarkPoolMarket

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
