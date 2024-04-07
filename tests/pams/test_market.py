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
