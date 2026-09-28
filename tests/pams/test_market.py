import copy
import math
import random
import time
from typing import List
from typing import Optional
from unittest import mock

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Cancel
from pams import Market
from pams import Order
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
        assert sum([log.volume for log in logs]) == 2
        execution_logs = [
            log for log in logger.pending_logs if isinstance(log, ExecutionLog)
        ]
        assert len(execution_logs) == len(logs)
        assert [id(log) for log in execution_logs] == [id(log) for log in logs]
        n_pending_logs = len(logger.pending_logs)
        assert market._execution() == []
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
        buys = [(None, buy_book[None])] if None in buy_book else []
        buys += sorted(
            [(p, v) for p, v in buy_book.items() if p is not None], reverse=True
        )
        sells = [(None, sell_book[None])] if None in sell_book else []
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
