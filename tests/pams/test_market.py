import copy
import math
import random
import time
import warnings
from typing import List
from typing import Optional
from typing import Tuple
from unittest import mock

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Cancel
from pams import Market
from pams import Order
from pams import OrderBook
from pams.logs.base import ExecutionLog
from pams.logs.base import ExpirationLog
from pams.logs.base import Logger
from pams.simulator import Simulator


class TestMarket:
    base_class = Market

    def test_init__(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m._update_time(next_fundamental_price=1.0)
        assert m._market_prices == [1.0] + [None for _ in range(m.chunk_size - 1)]
        assert m._last_executed_prices == [None for _ in range(m.chunk_size)]
        assert m._fundamental_prices == [1.0] + [None for _ in range(m.chunk_size - 1)]
        assert m._executed_volumes == [0 for _ in range(m.chunk_size)]
        assert m._executed_total_prices == [0.0 for _ in range(m.chunk_size)]
        assert m._n_buy_orders == [0 for _ in range(m.chunk_size)]
        assert m._n_sell_orders == [0 for _ in range(m.chunk_size)]
        assert m.get_market_price() == 1.0
        assert m.get_market_prices() == [1.0]
        assert m.get_last_executed_prices() == [None]
        assert m.get_last_executed_price() is None
        assert m.get_fundamental_prices() == [1.0]
        assert m.get_fundamental_price() == 1.0
        assert m.get_executed_volumes() == [0]
        assert m.get_executed_volume() == 0
        assert m.get_executed_total_prices() == [0]
        assert m.get_executed_total_price() == 0
        assert m.get_n_buy_orders() == [0]
        assert m.get_n_buy_order() == 0
        assert m.get_n_sell_orders() == [0]
        assert m.get_n_sell_order() == 0
        with pytest.raises(AssertionError):
            m.get_market_prices(range(2))
        with pytest.raises(AssertionError):
            m.get_market_price(1)
        with pytest.raises(AssertionError):
            m.get_last_executed_prices(range(2))
        with pytest.raises(AssertionError):
            m.get_last_executed_price(1)
        with pytest.raises(AssertionError):
            m.get_fundamental_prices(range(2))
        with pytest.raises(AssertionError):
            m.get_fundamental_price(1)
        with pytest.raises(AssertionError):
            m.get_executed_volumes(range(2))
        with pytest.raises(AssertionError):
            m.get_executed_volume(1)
        with pytest.raises(AssertionError):
            m.get_executed_total_prices(range(2))
        with pytest.raises(AssertionError):
            m.get_executed_total_price(1)
        with pytest.raises(AssertionError):
            m.get_n_buy_orders(range(2))
        with pytest.raises(AssertionError):
            m.get_n_buy_order(1)
        with pytest.raises(AssertionError):
            m.get_n_sell_orders(range(2))
        with pytest.raises(AssertionError):
            m.get_n_sell_order(1)
        with pytest.raises(AssertionError):
            m.get_vwap(1)
        assert math.isnan(m.get_vwap())
        m._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        m._add_order(order=order)
        assert order.order_id == 0
        m._execution()
        m._update_time(next_fundamental_price=1.1)
        order2 = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        m._add_order(order=order2)
        assert order2.order_id == 1
        m._execution()
        m._execution()
        m._update_time(next_fundamental_price=1.2)
        assert m._market_prices == [1.0, 1.0, 1.0] + [
            None for _ in range(m.chunk_size - 3)
        ]
        assert m._last_executed_prices == [None, 1.0, 1.0] + [
            None for _ in range(m.chunk_size - 3)
        ]
        assert m._fundamental_prices == [1.0, 1.1, 1.2] + [
            None for _ in range(m.chunk_size - 3)
        ]
        assert m._executed_volumes == [1 if i == 1 else 0 for i in range(m.chunk_size)]
        assert m._executed_total_prices == [
            1.0 if i == 1 else 0 for i in range(m.chunk_size)
        ]
        assert m._n_buy_orders == [1 if i == 0 else 0 for i in range(m.chunk_size)]
        assert m._n_sell_orders == [1 if i == 1 else 0 for i in range(m.chunk_size)]
        assert m.get_market_price() == 1.0
        assert m.get_market_prices() == [1.0, 1.0, 1.0]
        assert m.get_last_executed_prices() == [None, 1.0, 1.0]
        assert m.get_last_executed_price() == 1.0
        assert m.get_fundamental_prices() == [1.0, 1.1, 1.2]
        assert m.get_fundamental_price() == 1.2
        assert m.get_executed_volumes() == [0, 1, 0]
        assert m.get_executed_volume() == 0
        assert m.get_executed_total_prices() == [0, 1.0, 0]
        assert m.get_executed_total_price() == 0
        assert m.get_n_buy_orders() == [1, 0, 0]
        assert m.get_n_buy_order() == 0
        assert m.get_n_sell_orders() == [0, 1, 0]
        assert m.get_n_sell_order() == 0
        assert m.get_mid_prices() == [None, None, None]
        assert m.get_mid_price() is None
        assert m.get_vwap() == 1.0
        order3 = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.2
        )
        m._add_order(order=order3)
        order4 = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.1
        )
        m._add_order(order=order4)
        assert m.get_mid_prices() == [None, None, 1.5]  # because of tick size
        assert m.get_mid_price() == 1.5  # because of tick size
        assert m.get_sell_order_book() == {2.0: 1}
        assert m.get_buy_order_book() == {1.0: 1}
        assert m.convert_to_price(tick_level=2) == 2.0
        m._set_time(time=3, next_fundamental_price=1.3)

        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m._update_time(next_fundamental_price=1.0)
        m._is_running = True
        order3 = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.2
        )
        m._add_order(order=order3)
        order4 = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.1
        )
        m._add_order(order=order4)
        m._update_time(next_fundamental_price=1.2)
        m._set_time(time=2, next_fundamental_price=1.3)
        cancel_order = Cancel(order=order4)
        log = m._cancel_order(cancel_order)
        assert log is not None
        dummy_order = copy.deepcopy(order3)
        dummy_order.market_id = 1
        cancel_dummy = Cancel(order=dummy_order)
        with pytest.raises(ValueError):
            m._cancel_order(cancel_dummy)
        dummy_order = copy.deepcopy(order3)
        dummy_order.order_id = None
        cancel_dummy = Cancel(order=dummy_order)
        with pytest.raises(ValueError):
            m._cancel_order(cancel_dummy)
        dummy_order = copy.deepcopy(order3)
        dummy_order.placed_at = None
        cancel_dummy = Cancel(order=dummy_order)
        with pytest.raises(ValueError):
            m._cancel_order(cancel_dummy)
        cancel_dummy = Cancel(order=order3)
        with mock.patch("pams.order_book.OrderBook.cancel", return_value=None):
            with pytest.raises(AssertionError):
                m._cancel_order(cancel_dummy)

    @pytest.mark.parametrize(
        "history, expected",
        [
            ([None, None, None], None),
            ([1.0, None, None], 1.0),
            ([1.0, 2.0, None], 2.0),
            ([0.0, None, None], 0.0),
            ([None, 0.0, None], 0.0),
            ([1.0, -1.0, None], -1.0),
            ([-5.0, 5.0, None], 5.0),
        ],
    )
    def test_set_time_carries_last_prices(
        self, history: List[Optional[float]], expected: Optional[float]
    ) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m._update_time(next_fundamental_price=1.0)
        m._update_time(next_fundamental_price=1.0)
        m._last_executed_prices[: len(history)] = history
        m._mid_prices[: len(history)] = history
        m._market_prices[: len(history)] = history
        m._set_time(time=3, next_fundamental_price=1.0)
        assert m._last_executed_prices[3] == expected
        assert m._mid_prices[3] == expected
        assert m._market_prices[3] == expected

    @pytest.mark.parametrize(
        "executed, mid, expected",
        [
            ([None, 0.0], [None, 2.0], 0.0),
            ([None, None], [None, 0.0], 0.0),
            ([None, -1.0], [None, 2.0], -1.0),
        ],
    )
    def test_set_time_market_price_with_non_positive_prices(
        self,
        executed: List[Optional[float]],
        mid: List[Optional[float]],
        expected: Optional[float],
    ) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m._update_time(next_fundamental_price=1.0)
        m._update_time(next_fundamental_price=1.0)
        m._is_running = True
        m._last_executed_prices[:2] = executed
        m._mid_prices[:2] = mid
        m._market_prices[:2] = [1.0, 3.0]
        m._set_time(time=2, next_fundamental_price=1.0)
        assert m._market_prices[2] == expected

    def test_repr_(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        assert (
            str(m)
            == f"<{self.base_class.__module__}.{self.base_class.__name__} | id=0, name=test, tick_size=1.0,"
            f" outstanding_shares=None>"
        )

    def test_setup(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        with pytest.raises(ValueError):
            m.setup(
                settings={
                    "outstandingShares": 100,
                    "marketPrice": 300.0,
                    "fundamentalPrice": 500.0,
                }
            )
        with pytest.raises(ValueError):
            m.setup(
                settings={
                    "tickSize": 0.001,
                    "outstandingShares": 100.0,
                    "marketPrice": 300.0,
                    "fundamentalPrice": 500.0,
                }
            )
        with pytest.raises(ValueError):
            m.setup(settings={"tickSize": 0.001, "outstandingShares": 100})
        m.setup(
            settings={
                "tickSize": 0.001,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        m.setup(settings={"tickSize": 0.001, "fundamentalPrice": 500.0})
        m.setup(settings={"tickSize": 0.001, "marketPrice": 300.0})

    @pytest.mark.parametrize(
        "rate, expected", [(None, 0.0), (0.0, 0.0), (0, 0.0), (0.001, 0.001)]
    )
    def test_setup_transaction_cost_rate(
        self, rate: Optional[float], expected: float
    ) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        assert m.transaction_cost_rate == 0.0
        settings = {"tickSize": 0.001, "marketPrice": 300.0}
        if rate is not None:
            settings["transactionCostRate"] = rate
        m.setup(settings=settings)
        assert m.transaction_cost_rate == expected
        assert isinstance(m.transaction_cost_rate, float)

    @pytest.mark.parametrize(
        "rate, match",
        [
            ("0.001", "must be int or float"),
            (True, "must be int or float"),
            (False, "must be int or float"),
            (None, "must be int or float"),
            (-0.001, r"must be in \[0.0, 1.0\)"),
            (1.0, r"must be in \[0.0, 1.0\)"),
            (1, r"must be in \[0.0, 1.0\)"),
            (10.0, r"must be in \[0.0, 1.0\)"),
            (math.nan, r"must be in \[0.0, 1.0\)"),
            (math.inf, r"must be in \[0.0, 1.0\)"),
        ],
    )
    def test_setup_transaction_cost_rate_invalid(
        self, rate: object, match: str
    ) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        with pytest.raises(ValueError, match=match):
            m.setup(
                settings={
                    "tickSize": 0.001,
                    "marketPrice": 300.0,
                    "transactionCostRate": rate,
                }
            )

    def test_extract_sequential_data_by_time(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        with pytest.raises(AssertionError):
            m._extract_sequential_data_by_time(
                times=[1, 2], parameters=[0, 1, 2, 3], allow_none=False
            )
        m.time = 1
        with pytest.raises(AssertionError):
            m._extract_sequential_data_by_time(
                times=[1, 2], parameters=[0, 1, 2, 3], allow_none=False
            )
        m.time = 2
        m._extract_sequential_data_by_time(
            times=[1, 2], parameters=[0, 1, 2, 3], allow_none=False
        )
        m.time = 4
        results: List[Optional[int]] = m._extract_sequential_data_by_time(
            times=[1, 2], parameters=[0, 1, 2, 3], allow_none=False
        )
        expected: List[Optional[int]] = [1, 2]
        assert results == expected
        results = m._extract_sequential_data_by_time(
            times=[1, 2], parameters=[0, 1, 2, 3], allow_none=True
        )
        expected = [1, 2]
        assert results == expected
        with pytest.raises(AssertionError):
            m._extract_sequential_data_by_time(
                times=[1, 2], parameters=[0, 1, None, 3], allow_none=False
            )
        results = m._extract_sequential_data_by_time(
            times=[1, 2], parameters=[0, 1, None, 3], allow_none=True
        )
        expected = [1, None]
        assert results == expected

    def test_extract_sequential_data_by_time_with_iterators(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m.time = 4
        parameters: List[Optional[int]] = [0, 1, 2, 3, 4]
        assert m._extract_sequential_data_by_time(
            times=(t for t in [1, 2]), parameters=parameters
        ) == [1, 2]
        assert m._extract_sequential_data_by_time(
            times=iter([3, 0, 3, 1]), parameters=parameters
        ) == [3, 0, 3, 1]
        assert m._extract_sequential_data_by_time(
            times=range(1, 4), parameters=parameters
        ) == [1, 2, 3]
        assert (
            m._extract_sequential_data_by_time(times=iter([]), parameters=parameters)
            == []
        )
        with pytest.raises(AssertionError):
            m._extract_sequential_data_by_time(
                times=(t for t in [1, 5]), parameters=parameters + [5]
            )
        m._market_prices = [1.0, 1.1, 1.2, 1.3, 1.4]
        assert m.get_market_prices(iter([0])) == [1.0]
        assert m.get_market_prices(t for t in [4, 2]) == [1.4, 1.2]

    def test_extract_data_by_time(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        with pytest.raises(AssertionError):
            m._extract_data_by_time(time=1, parameters=[0, 1, 2, 3], allow_none=False)
        m.time = 1
        result: Optional[int] = m._extract_data_by_time(
            time=1, parameters=[0, 1, 2, 3], allow_none=False
        )
        expected: Optional[int] = 1
        assert result == expected
        m.time = 3
        result = m._extract_data_by_time(
            time=1, parameters=[0, 1, 2, 3], allow_none=True
        )
        expected = 1
        assert result == expected
        with pytest.raises(AssertionError):
            m._extract_data_by_time(
                time=2, parameters=[0, 1, None, 3], allow_none=False
            )
        result = m._extract_data_by_time(
            time=2, parameters=[0, 1, None, 3], allow_none=True
        )
        expected = None
        assert result == expected

    def test_get_time(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        assert m.get_time() == -1
        m._update_time(next_fundamental_price=300)
        assert m.get_time() == 0
        m.time = 3
        assert m.get_time() == 3

    def test_execute_orders(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m._update_time(next_fundamental_price=1.0)
        order_sell = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        m._add_order(order=order_sell)
        order_buy = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=2.0
        )
        m._add_order(order=order_buy)
        with pytest.raises(AssertionError):
            m._execute_orders(
                price=10, volume=1, buy_order=order_buy, sell_order=order_sell
            )
        m._is_running = True
        order_buy.market_id = 1
        with pytest.raises(ValueError):
            m._execute_orders(
                price=10, volume=1, buy_order=order_buy, sell_order=order_sell
            )
        order_buy.market_id = 0
        order_sell.market_id = 1
        with pytest.raises(ValueError):
            m._execute_orders(
                price=10, volume=1, buy_order=order_buy, sell_order=order_sell
            )
        order_sell.market_id = 0
        with pytest.raises(AssertionError):
            m._execute_orders(
                price=1.5, volume=0, buy_order=order_buy, sell_order=order_sell
            )
        m._execute_orders(
            price=1.5, volume=1, buy_order=order_buy, sell_order=order_sell
        )
        order_sell.placed_at = None
        with pytest.raises(ValueError):
            m._execute_orders(
                price=1.5, volume=1, buy_order=order_buy, sell_order=order_sell
            )
        order_buy.placed_at = None
        with pytest.raises(ValueError):
            m._execute_orders(
                price=1.5, volume=1, buy_order=order_buy, sell_order=order_sell
            )

    def test_add_order(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m._update_time(next_fundamental_price=1.0)
        order_sell = Order(
            agent_id=0, market_id=1, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        with pytest.raises(ValueError):
            m._add_order(order=order_sell)
        order_sell = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            placed_at=0,
        )
        with pytest.raises(ValueError):
            m._add_order(order=order_sell)
        order_sell = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        with pytest.raises(ValueError):
            m._add_order(order=order_sell)
        with mock.patch("pams.order_book.OrderBook.add", return_value=None):
            order_sell = Order(
                agent_id=0,
                market_id=0,
                is_buy=False,
                kind=LIMIT_ORDER,
                volume=1,
                price=1.0,
            )
            with pytest.raises(AssertionError):
                m._add_order(order=order_sell)

    def _create_market_with_tick_size(self, tick_size: float) -> Market:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m.tick_size = tick_size
        m._update_time(next_fundamental_price=1.0)
        return m

    @pytest.mark.parametrize(
        "tick_size, tick_level",
        [
            (1.0, 3),
            (1.0, 123456789),
            (1.0, 500000000001),
            (0.1, 3),
            (0.1, 7),
            (0.1, 12345),
            (0.01, 29),
            (0.01, 57),
            (0.01, 1234567),
            (0.00001, 30000),
            (0.00001, 70001),
            (0.00001, 10000000),
            (0.00001, 123456789),
            (0.00001, 500000000001),
        ],
    )
    @pytest.mark.parametrize("is_buy", [True, False])
    def test_add_order_on_tick_price(
        self, tick_size: float, tick_level: int, is_buy: bool
    ) -> None:
        m = self._create_market_with_tick_size(tick_size=tick_size)
        # prices written as decimal literals, as users / configs typically do
        decimals = max(0, -math.floor(math.log10(tick_size)))
        prices = [
            float(f"{tick_level * tick_size:.{decimals}f}"),
            tick_level * tick_size,
            m.convert_to_price(tick_level=tick_level),
        ]
        for price in prices:
            assert m.convert_to_tick_level_rounded_lower(price=price) == tick_level
            assert m.convert_to_tick_level_rounded_upper(price=price) == tick_level
            assert m.convert_to_tick_level(price=price, is_buy=is_buy) == tick_level
            order = Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=1,
                price=price,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                m._add_order(order=order)
            assert order.price == m.convert_to_price(tick_level=tick_level)
            # idempotence: converting the modified price again is stable
            assert order.price is not None
            for side in [True, False]:
                assert (
                    m.convert_to_tick_level(price=order.price, is_buy=side)
                    == tick_level
                )

    @pytest.mark.parametrize(
        "tick_size, price, lower, upper",
        [
            (1.0, 1.1, 1, 2),
            (1.0, 2.5, 2, 3),
            (1.0, 3.000001, 3, 4),
            (1.0, 1000000000.5, 1000000000, 1000000001),
            (1.0, 500000000001.5, 500000000001, 500000000002),
            (1.0, 500000000001.001, 500000000001, 500000000002),
            (1.0, 500000000001.0005, 500000000001, 500000000002),
            (1.0, 499999999999.999, 499999999999, 500000000000),
            (1.0, 123456789012.001, 123456789012, 123456789013),
            (1.0, 2.0**49 + 0.5, 2**49, 2**49 + 1),
            (1.0, -2.5, -3, -2),
            (0.1, 0.35, 3, 4),
            (0.1, 0.71, 7, 8),
            (0.01, 0.295, 29, 30),
            (0.01, 1.2345, 123, 124),
            (0.01, 1234567.8900001, 123456789, 123456790),
            (0.00001, 1.000005, 100000, 100001),
            (0.00001, 0.300001, 30000, 30001),
            (0.00001, 1234.567891, 123456789, 123456790),
            (0.00001, 1000.000000001, 100000000, 100000001),
        ],
    )
    @pytest.mark.parametrize("is_buy", [True, False])
    def test_add_order_off_tick_price(
        self, tick_size: float, price: float, lower: int, upper: int, is_buy: bool
    ) -> None:
        m = self._create_market_with_tick_size(tick_size=tick_size)
        assert m.convert_to_tick_level_rounded_lower(price=price) == lower
        assert m.convert_to_tick_level_rounded_upper(price=price) == upper
        expected_level = lower if is_buy else upper
        assert m.convert_to_tick_level(price=price, is_buy=is_buy) == expected_level
        order = Order(
            agent_id=0,
            market_id=0,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            volume=1,
            price=price,
        )
        with pytest.warns(UserWarning):
            m._add_order(order=order)
        assert order.price == m.convert_to_price(tick_level=expected_level)
        # the modified price is on tick, so converting it again is stable
        assert order.price is not None
        for side in [True, False]:
            assert (
                m.convert_to_tick_level(price=order.price, is_buy=side)
                == expected_level
            )

    def test_add_order_on_tick_price_same_level(self) -> None:
        m = self._create_market_with_tick_size(tick_size=0.1)
        m._is_running = True
        for price in [0.3, 3 * 0.1, 0.1 + 0.2]:
            order = Order(
                agent_id=0,
                market_id=0,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=price,
            )
            m._add_order(order=order)
        assert m.get_buy_order_book() == {m.convert_to_price(tick_level=3): 3}
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=3, price=0.3
        )
        m._add_order(order=order)
        logs = m._execution()
        assert sum(log.volume for log in logs) == 3

    def test_execution(self) -> None:
        random.seed(42)
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        for _ in range(100):
            kind = LIMIT_ORDER if random.random() < 0.7 else MARKET_ORDER
            price = random.random() * 10 if kind == LIMIT_ORDER else None
            ttl = random.randint(1, 100) if bool(random.getrandbits(1)) else None
            order = Order(
                agent_id=0,
                market_id=0,
                is_buy=bool(random.getrandbits(1)),
                kind=kind,
                volume=random.randint(1, 10),
                price=price,
                ttl=ttl,
            )
            market._add_order(order)
            market._update_time(1.0)
        market._is_running = True
        logs = market._execution()
        if len(logs) == 0:
            raise AssertionError
        start_time = time.time()
        n_logs = 0
        total_volume = 0
        for _ in range(5000):
            kind = LIMIT_ORDER if random.random() < 0.6 else MARKET_ORDER
            price = random.random() * 10 if kind == LIMIT_ORDER else None
            ttl = random.randint(1, 100) if bool(random.getrandbits(1)) else None
            volume = random.randint(1, 10)
            total_volume += volume
            order = Order(
                agent_id=0,
                market_id=0,
                is_buy=bool(random.getrandbits(1)),
                kind=kind,
                volume=volume,
                price=price,
                ttl=ttl,
            )
            market._add_order(order)
            log = market._execution()
            n_logs += len(log)
            market._update_time(1.0)
        assert n_logs <= total_volume
        end_time = time.time()
        time_per_step = (end_time - start_time) / 10000
        print("time/step", time_per_step)
        assert time_per_step < 0.005

    def test_execution_order_pattern1(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        assert not market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 0

    def test_execution_order_pattern2(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        assert not market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 0

    def test_execution_order_pattern3(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=1
        )
        market._add_order(order)
        order = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        market._add_order(order)
        assert not market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 0

    def test_execution_order_pattern4(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=1
        )
        market._add_order(order)
        order = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 2

    def test_execution_order_pattern5(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=2, price=8
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 2

    def test_execution_order_pattern6(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=8
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=2, price=11
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 2

    def test_execution_order_pattern7(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=8
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=11
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 1

    def test_execution_order_pattern8(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=1
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=11
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 1

    def test_execution_order_pattern9(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=8
        )
        market._add_order(order)
        order = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 1

    def test_execution_order_pattern10(self) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=2
        )
        market._add_order(order)
        order = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 2

    def test_execution_logs_written_once(self) -> None:
        logger = Logger()
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=logger,
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=9
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(order)
        order = Order(
            agent_id=1, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=2, price=10
        )
        market._add_order(order)
        logs = market._execution()
        assert len(logs) == 2
        assert all(isinstance(log, ExecutionLog) for log in logs)
        assert sum(log.volume for log in logs) == 2
        execution_logs = [
            log for log in logger.pending_logs if isinstance(log, ExecutionLog)
        ]
        assert len(execution_logs) == len(logs)
        assert [id(log) for log in execution_logs] == [id(log) for log in logs]
        n_pending_logs = len(logger.pending_logs)
        assert not market._execution()
        assert len(logger.pending_logs) == n_pending_logs

    def test_execute_orders_log_written_once(self) -> None:
        logger = Logger()
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=logger,
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        sell_order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(sell_order)
        buy_order = Order(
            agent_id=1, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10
        )
        market._add_order(buy_order)
        log = market._execute_orders(
            price=10.0, volume=1, buy_order=buy_order, sell_order=sell_order
        )
        execution_logs = [
            log_ for log_ in logger.pending_logs if isinstance(log_, ExecutionLog)
        ]
        assert execution_logs == [log]

    def test_compute_transaction_costs(self) -> None:
        market = self._make_running_market()
        buy_order = Order(
            agent_id=1, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=3, price=10
        )
        sell_order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=3, price=10
        )
        assert market.compute_transaction_costs(
            price=10.0, volume=3, buy_order=buy_order, sell_order=sell_order
        ) == (0.0, 0.0)
        market.transaction_cost_rate = 0.001
        buy_cost, sell_cost = market.compute_transaction_costs(
            price=10.0, volume=3, buy_order=buy_order, sell_order=sell_order
        )
        assert buy_cost == pytest.approx(0.03)
        assert sell_cost == pytest.approx(0.03)

    @pytest.mark.parametrize("rate", [None, 0.0, 0.002])
    def test_execute_orders_transaction_costs(self, rate: Optional[float]) -> None:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        settings = {"tickSize": 0.001, "marketPrice": 10.0}
        if rate is not None:
            settings["transactionCostRate"] = rate
        market.setup(settings=settings)
        market._update_time(10.0)
        market._is_running = True
        sell_order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=5, price=10
        )
        market._add_order(sell_order)
        buy_order = Order(
            agent_id=1, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=2, price=10
        )
        market._add_order(buy_order)
        log = market._execute_orders(
            price=10.0, volume=2, buy_order=buy_order, sell_order=sell_order
        )
        assert log.price == 10.0
        assert log.volume == 2
        expected = 0.0 if rate is None else 10.0 * 2 * rate
        assert log.buy_transaction_cost == pytest.approx(expected)
        assert log.sell_transaction_cost == pytest.approx(expected)
        if not expected:
            assert log.buy_transaction_cost == 0.0
            assert log.sell_transaction_cost == 0.0

    @pytest.mark.parametrize("buy_first", [False, True])
    def test_execute_orders_overridden_transaction_costs(self, buy_first: bool) -> None:
        calls: List[Tuple[float, int, Order, Order]] = []

        class MakerTakerMarket(self.base_class):  # type: ignore
            def compute_transaction_costs(
                self, price: float, volume: int, buy_order: Order, sell_order: Order
            ) -> Tuple[float, float]:
                calls.append((price, volume, buy_order, sell_order))
                # The order that arrived first (smaller order ID) is the maker.
                assert buy_order.order_id is not None
                assert sell_order.order_id is not None
                value = price * volume
                maker_cost = -0.0001 * value
                taker_cost = 0.0003 * value
                if buy_order.order_id > sell_order.order_id:
                    return taker_cost, maker_cost
                return maker_cost, taker_cost

        market = MakerTakerMarket(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        sell_order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=2, price=10
        )
        buy_order = Order(
            agent_id=1, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=2, price=10
        )
        if buy_first:
            market._add_order(buy_order)
            market._add_order(sell_order)
        else:
            market._add_order(sell_order)
            market._add_order(buy_order)
        logs = market._execution()
        assert len(logs) == 1
        assert calls == [(10.0, 2, buy_order, sell_order)]
        # The maker gets a rebate of 0.002 and the taker pays 0.006.
        maker_cost, taker_cost = -0.002, 0.006
        if buy_first:
            assert logs[0].buy_transaction_cost == pytest.approx(maker_cost)
            assert logs[0].sell_transaction_cost == pytest.approx(taker_cost)
        else:
            assert logs[0].buy_transaction_cost == pytest.approx(taker_cost)
            assert logs[0].sell_transaction_cost == pytest.approx(maker_cost)

    def _make_running_market(self) -> Market:
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        return market

    @pytest.mark.parametrize("is_buy_larger", [True, False])
    def test_execution_order_unequal_market_volumes(self, is_buy_larger: bool) -> None:
        # larger market side: 10, smaller market side: 1 + one limit level of 100
        market = self._make_running_market()
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=MARKET_ORDER,
                volume=10,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=MARKET_ORDER,
                volume=1,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=LIMIT_ORDER,
                volume=100,
                price=300,
            )
        )
        assert market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 2
        assert sum(log.volume for log in logs) == 10
        assert all(log.price == 300 for log in logs)
        assert not market.remain_executable_orders()

    @pytest.mark.parametrize("is_buy_larger", [True, False])
    @pytest.mark.parametrize(
        "limit_volumes, expected",
        [
            ([3, 3, 3], True),  # excess 9 == total limit volume over 3 levels
            ([2, 2, 2], False),  # excess 9 > total limit volume 6
            ([9], True),
            ([8], False),
        ],
    )
    def test_execution_order_unequal_market_volumes_limit_volume(
        self, is_buy_larger: bool, limit_volumes: List[int], expected: bool
    ) -> None:
        # larger market side: 10, smaller market side: 1 + limit orders
        market = self._make_running_market()
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=MARKET_ORDER,
                volume=10,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=MARKET_ORDER,
                volume=1,
            )
        )
        for i, volume in enumerate(limit_volumes):
            market._add_order(
                Order(
                    agent_id=0,
                    market_id=0,
                    is_buy=not is_buy_larger,
                    kind=LIMIT_ORDER,
                    volume=volume,
                    price=300 + i,
                )
            )
        assert market.remain_executable_orders() == expected
        logs = market._execution()
        assert sum(log.volume for log in logs) == (10 if expected else 0)
        # all executions of a batch share the price of the last matched level
        last_price = 300 + len(limit_volumes) - 1 if is_buy_larger else 300
        assert all(log.price == last_price for log in logs)
        larger_book = market.buy_order_book if is_buy_larger else market.sell_order_book
        assert len(larger_book) == (0 if expected else 1)
        assert not market.remain_executable_orders()

    @pytest.mark.parametrize("is_buy_larger", [True, False])
    @pytest.mark.parametrize("smaller_limit_volume, expected", [(4, False), (9, True)])
    def test_execution_order_unequal_market_volumes_larger_side_limit(
        self, is_buy_larger: bool, smaller_limit_volume: int, expected: bool
    ) -> None:
        # larger market side: 10 + crossing limit 5, smaller market side: 1 + limit.
        # Only the limit orders of the smaller side can absorb the excess volume 9.
        market = self._make_running_market()
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=MARKET_ORDER,
                volume=10,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=LIMIT_ORDER,
                volume=5,
                price=400 if is_buy_larger else 200,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=MARKET_ORDER,
                volume=1,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=LIMIT_ORDER,
                volume=smaller_limit_volume,
                price=300,
            )
        )
        assert market.remain_executable_orders() == expected
        logs = market._execution()
        larger_book = market.buy_order_book if is_buy_larger else market.sell_order_book
        smaller_book = (
            market.sell_order_book if is_buy_larger else market.buy_order_book
        )
        if expected:
            assert sum(log.volume for log in logs) == 10
            assert all(log.price == 300 for log in logs)
            assert larger_book.get_price_volume() == {400 if is_buy_larger else 200: 5}
            assert len(smaller_book) == 0
        else:
            assert len(logs) == 0
            assert larger_book.get_price_volume() == {
                None: 10,
                400 if is_buy_larger else 200: 5,
            }
            assert smaller_book.get_price_volume() == {None: 1, 300: 4}
        assert not market.remain_executable_orders()

    @pytest.mark.parametrize("is_buy_market", [True, False])
    def test_execution_order_one_sided_market_orders(self, is_buy_market: bool) -> None:
        # the special quote only applies when both sides have market orders.
        # market orders on one side are executed against the limit orders as far as
        # possible and the rest remains in the book.
        market = self._make_running_market()
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_market,
                kind=MARKET_ORDER,
                volume=10,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_market,
                kind=LIMIT_ORDER,
                volume=1,
                price=300,
            )
        )
        assert market.remain_executable_orders()
        logs = market._execution()
        assert [(log.volume, log.price) for log in logs] == [(1, 300)]
        market_book = market.buy_order_book if is_buy_market else market.sell_order_book
        assert market_book.get_price_volume() == {None: 9}
        assert not market.remain_executable_orders()

    @pytest.mark.parametrize("is_buy_larger", [True, False])
    def test_execution_order_unequal_market_volumes_special_quote(
        self, is_buy_larger: bool
    ) -> None:
        # larger market side: 10, smaller market side: 5 + limit volume 1.
        # The excess market volume 5 cannot be absorbed by the limit volume 1,
        # so the equilibrium price cannot be determined (special quote).
        market = self._make_running_market()
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=MARKET_ORDER,
                volume=10,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=MARKET_ORDER,
                volume=5,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=LIMIT_ORDER,
                volume=1,
                price=300,
            )
        )
        assert not market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 0
        assert not market.remain_executable_orders()

    @pytest.mark.parametrize("is_buy_larger", [True, False])
    def test_execution_order_unequal_market_volumes_no_limit(
        self, is_buy_larger: bool
    ) -> None:
        # the smaller market side has no limit order, so the excess market volume 9
        # cannot be absorbed (special quote).
        market = self._make_running_market()
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=MARKET_ORDER,
                volume=10,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy_larger,
                kind=LIMIT_ORDER,
                volume=5,
                price=300,
            )
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=not is_buy_larger,
                kind=MARKET_ORDER,
                volume=1,
            )
        )
        assert not market.remain_executable_orders()
        logs = market._execution()
        assert len(logs) == 0

    @pytest.mark.parametrize(
        "buy_price, sell_price, expected",
        [(10, 10, True), (10, 9, True), (9, 10, False), (10, None, False)],
    )
    def test_execution_order_equal_market_volumes(
        self, buy_price: int, sell_price: Optional[int], expected: bool
    ) -> None:
        market = self._make_running_market()
        market._add_order(
            Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=3)
        )
        market._add_order(
            Order(agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=3)
        )
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=buy_price,
            )
        )
        if sell_price is not None:
            market._add_order(
                Order(
                    agent_id=0,
                    market_id=0,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=sell_price,
                )
            )
        assert market.remain_executable_orders() == expected
        logs = market._execution()
        assert sum(log.volume for log in logs) == (4 if expected else 0)
        assert not market.remain_executable_orders()

    @staticmethod
    def _is_executable_reference(market: Market) -> bool:
        # walk both books in priority order (market orders first) like _execution
        # until the books do not cross anymore. The orders are executable if a price
        # is determined by a limit order and, when both sides have market orders,
        # all the market orders are matched (otherwise, special quote).
        buy_book = market.buy_order_book.get_price_volume()
        sell_book = market.sell_order_book.get_price_volume()
        both_market = None in buy_book and None in sell_book
        buys: List[Tuple[Optional[float], int]] = (
            [(None, buy_book[None])] if None in buy_book else []
        )
        buys += sorted(
            [(p, v) for p, v in buy_book.items() if p is not None], reverse=True
        )
        sells: List[Tuple[Optional[float], int]] = (
            [(None, sell_book[None])] if None in sell_book else []
        )
        sells += sorted([(p, v) for p, v in sell_book.items() if p is not None])
        i, j, buy_volume, sell_volume = 0, 0, 0, 0
        buy_price: Optional[float] = None
        sell_price: Optional[float] = None
        price_determined = False
        while True:
            if buy_volume == 0:
                if i == len(buys):
                    break
                buy_price, buy_volume = buys[i]
                i += 1
            if sell_volume == 0:
                if j == len(sells):
                    break
                sell_price, sell_volume = sells[j]
                j += 1
            if (
                buy_price is not None
                and sell_price is not None
                and buy_price < sell_price
            ):
                break
            if buy_price is not None or sell_price is not None:
                price_determined = True
            volume = min(buy_volume, sell_volume)
            buy_volume -= volume
            sell_volume -= volume
        market_left = (buy_price is None and buy_volume > 0) or (
            sell_price is None and sell_volume > 0
        )
        return price_determined and not (both_market and market_left)

    def test_remain_executable_orders_without_market_order_volume(self) -> None:
        sim = Simulator(prng=random.Random(42))
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        market_order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1
        )
        # both best orders are market orders but the order books report no
        # market-order volume: this inconsistency is rejected.
        with mock.patch("pams.order_book.OrderBook.__len__", return_value=1):
            with mock.patch(
                "pams.order_book.OrderBook.get_best_order", return_value=market_order
            ):
                with mock.patch(
                    "pams.order_book.OrderBook.get_price_volume", return_value={}
                ):
                    with pytest.raises(AssertionError):
                        market.remain_executable_orders()

    def test_remain_executable_orders_random(self) -> None:
        prng = random.Random(42)
        n_executed = 0
        for _ in range(2000):
            market = self._make_running_market()
            for _ in range(prng.randint(1, 8)):
                kind = MARKET_ORDER if prng.random() < 0.5 else LIMIT_ORDER
                market._add_order(
                    Order(
                        agent_id=0,
                        market_id=0,
                        is_buy=bool(prng.getrandbits(1)),
                        kind=kind,
                        volume=prng.randint(1, 10),
                        price=prng.randint(1, 3) if kind == LIMIT_ORDER else None,
                    )
                )
            buy_book = market.get_buy_order_book()
            sell_book = market.get_sell_order_book()
            executable = self._is_executable_reference(market)
            assert market.remain_executable_orders() == executable
            logs = market._execution()
            if executable:
                assert len(logs) > 0
                if None in buy_book and None in sell_book:
                    # no market order is left after execution (no special quote)
                    assert None not in market.get_buy_order_book()
                    assert None not in market.get_sell_order_book()
            else:
                assert len(logs) == 0
                assert market.get_buy_order_book() == buy_book
                assert market.get_sell_order_book() == sell_book
            assert not market.remain_executable_orders()
            assert not self._is_executable_reference(market)
            n_executed += int(executable)
        assert 0 < n_executed < 2000

    @staticmethod
    def _make_placed_order(
        is_buy: bool, price: Optional[float], placed_at: int, order_id: Optional[int]
    ) -> Order:
        return Order(
            agent_id=0,
            market_id=0,
            is_buy=is_buy,
            kind=MARKET_ORDER if price is None else LIMIT_ORDER,
            volume=1,
            placed_at=placed_at,
            price=price,
            order_id=order_id,
        )

    @pytest.mark.parametrize(
        "buy, sell, expected",
        [
            # (price, placed_at, order_id) of the buy and sell orders
            ((None, 1, 0), (None, 1, 1), None),  # both market orders
            ((None, 1, 0), (90.0, 2, 1), 90.0),  # market buy
            ((110.0, 2, 0), (None, 1, 1), 110.0),  # market sell
            ((110.0, 1, 0), (90.0, 2, 1), 110.0),  # buy placed earlier
            ((110.0, 2, 0), (90.0, 1, 1), 90.0),  # sell placed earlier
            ((110.0, 1, 0), (90.0, 1, 1), 110.0),  # same time, buy has smaller ID
            ((110.0, 1, 1), (90.0, 1, 0), 90.0),  # same time, sell has smaller ID
        ],
    )
    def test_get_execution_price(
        self,
        buy: Tuple[Optional[float], int, int],
        sell: Tuple[Optional[float], int, int],
        expected: Optional[float],
    ) -> None:
        buy_order = self._make_placed_order(True, *buy)
        sell_order = self._make_placed_order(False, *sell)
        assert (
            self.base_class._get_execution_price(
                buy_order=buy_order, sell_order=sell_order
            )
            == expected
        )

    @pytest.mark.parametrize(
        "buy_order_id, sell_order_id", [(None, 1), (0, None), (0, 0)]
    )
    def test_get_execution_price_same_time_invalid_order_ids(
        self, buy_order_id: Optional[int], sell_order_id: Optional[int]
    ) -> None:
        # limit orders placed at the same time must have assigned, distinct order IDs
        buy_order = self._make_placed_order(
            is_buy=True, price=110.0, placed_at=1, order_id=buy_order_id
        )
        sell_order = self._make_placed_order(
            is_buy=False, price=90.0, placed_at=1, order_id=sell_order_id
        )
        with pytest.raises(AssertionError):
            self.base_class._get_execution_price(
                buy_order=buy_order, sell_order=sell_order
            )

    def test_pop_next_order(self) -> None:
        order_book = OrderBook(is_buy=True)
        popped_orders: List[Order] = []
        assert (
            self.base_class._pop_next_order(
                order_book=order_book, popped_orders=popped_orders
            )
            is None
        )
        assert not popped_orders

        # OrderBook.add sets placed_at to the time of the order book (0)
        orders = [
            self._make_placed_order(
                is_buy=True, price=price, placed_at=0, order_id=order_id
            )
            for order_id, price in enumerate([90.0, 110.0, 100.0])
        ]
        for order in orders:
            order_book.add(order)
        # the best buy orders are popped first and the heap order is kept
        for expected in [orders[1], orders[2], orders[0]]:
            assert (
                self.base_class._pop_next_order(
                    order_book=order_book, popped_orders=popped_orders
                )
                is expected
            )
        assert popped_orders == [orders[1], orders[2], orders[0]]
        assert not order_book.priority_queue

    def test_pop_next_order_without_volume(self) -> None:
        order_book = OrderBook(is_buy=True)
        order = self._make_placed_order(
            is_buy=True, price=100.0, placed_at=0, order_id=0
        )
        order_book.add(order)
        # an order without volume must not remain in the order book
        order.volume = 0
        popped_orders: List[Order] = []
        with pytest.raises(AssertionError):
            self.base_class._pop_next_order(
                order_book=order_book, popped_orders=popped_orders
            )
        # the popped order is still recorded so that it can be pushed back
        assert popped_orders == [order]
        assert not order_book.priority_queue

    def _make_market_with_best_order_without_volume(
        self, is_buy: bool
    ) -> Tuple[Market, Order, Order]:
        # The best order of one side has no volume, while the next order of that side
        # still crosses the order of the other side.
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        sign = 1 if is_buy else -1
        order_without_volume = Order(
            agent_id=0,
            market_id=0,
            is_buy=is_buy,
            kind=LIMIT_ORDER,
            volume=1,
            price=100 + 10 * sign,
        )
        market._add_order(order_without_volume)
        market._add_order(
            Order(
                agent_id=0,
                market_id=0,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=10,
                price=100 + 5 * sign,
            )
        )
        other_order = Order(
            agent_id=0,
            market_id=0,
            is_buy=not is_buy,
            kind=LIMIT_ORDER,
            volume=5,
            price=100,
        )
        market._add_order(other_order)
        # an order without volume must not remain in the order book
        order_without_volume.volume = 0
        assert market.remain_executable_orders()
        return market, order_without_volume, other_order

    @pytest.mark.parametrize("is_buy", [True, False])
    def test_execution_best_order_without_volume(self, is_buy: bool) -> None:
        # an order without volume is not skipped even if it is the first popped order
        # of its side, but violates the invariant like the following ones
        market, _, _ = self._make_market_with_best_order_without_volume(is_buy=is_buy)
        with pytest.raises(AssertionError):
            market._execution()

    @pytest.mark.parametrize("is_buy", [True, False])
    def test_collect_pending_executions_best_order_without_volume(
        self, is_buy: bool
    ) -> None:
        (
            market,
            order_without_volume,
            other_order,
        ) = self._make_market_with_best_order_without_volume(is_buy=is_buy)
        popped_buy_orders: List[Order] = []
        popped_sell_orders: List[Order] = []
        with pytest.raises(AssertionError):
            market._collect_pending_executions(
                popped_buy_orders=popped_buy_orders,
                popped_sell_orders=popped_sell_orders,
            )
        # the buy order is popped first, and the popped orders are recorded so that
        # they can be pushed back
        if is_buy:
            assert popped_buy_orders == [order_without_volume]
            assert not popped_sell_orders
        else:
            assert popped_buy_orders == [other_order]
            assert popped_sell_orders == [order_without_volume]

    def test_expiration_orrder_pattern01(self) -> None:
        logger = Logger()
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=logger,
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=2, ttl=1
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1, ttl=1
        )
        market._add_order(order)
        market._update_time(1.0)
        market._update_time(1.0)
        assert (
            len([log for log in logger.pending_logs if isinstance(log, ExpirationLog)])
            == 2
        )

    def test_expiration_orrder_pattern02(self) -> None:
        logger = Logger()
        market = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=logger,
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        market._update_time(1.0)
        market._is_running = True
        order = Order(
            agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=2, ttl=1
        )
        market._add_order(order)
        order = Order(
            agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1, ttl=1
        )
        market._add_order(order)
        market._set_time(time=2, next_fundamental_price=1.0)
        assert (
            len([log for log in logger.pending_logs if isinstance(log, ExpirationLog)])
            == 2
        )
