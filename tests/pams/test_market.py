import math
import random
import time
from typing import List
from typing import Optional

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Market
from pams import Order
from pams.logs.base import Logger
from pams.simulator import Simulator


class TestMarket:
    def test_init__(self) -> None:
        m = Market(
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

    def test_setup(self) -> None:
        m = Market(
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
        m = Market(
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
        m = Market(
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
        m = Market(
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

    def test_execution(self) -> None:
        random.seed(42)
        market = Market(
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
        for _ in range(5000):
            kind = LIMIT_ORDER if random.random() < 0.7 else MARKET_ORDER
            price = random.random() * 10 if kind == LIMIT_ORDER else None
            ttl = random.randint(1, 100) if bool(random.getrandbits(1)) else None
            volume = random.randint(1, 10)
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
            assert len(log) <= volume
            market._update_time(1.0)
        end_time = time.time()
        time_per_step = (end_time - start_time) / 10000
        print("time/step", time_per_step)
        assert time_per_step < 0.005
