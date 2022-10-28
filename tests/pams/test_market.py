import math
import random
import time

import pytest

from pams import LIMIT_ORDER
from pams import MARKET_ORDER
from pams import Market
from pams import Order
from pams.logs import Logger
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
        for _ in range(1000):
            kind = LIMIT_ORDER if random.random() < 0.1 else MARKET_ORDER
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
        for _ in range(10000):
            kind = LIMIT_ORDER if random.random() < 0.1 else MARKET_ORDER
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
        assert time_per_step < 0.002
