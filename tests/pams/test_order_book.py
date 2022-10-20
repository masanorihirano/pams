import pytest

from pams import MARKET_ORDER
from pams import Cancel
from pams import Order
from pams import OrderBook


class TestOrderBook:
    def test_init(self) -> None:
        ob = OrderBook(is_buy=True)
        o = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        ob.add(o)
        o = Order(agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=1)
        with pytest.raises(ValueError):
            ob.add(o)
        assert len(ob) == 1
        ob = OrderBook(is_buy=False)
        o = Order(agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=1)
        c = Cancel(order=o)
        ob.add(o)
        o2 = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        with pytest.raises(ValueError):
            ob.add(o2)
        assert len(ob) == 1
        assert ob.get_best_order() == o
        assert ob.get_best_price() is None
        ob.cancel(c)
        assert len(ob) == 0
        assert ob.get_best_order() is None
        assert ob.get_best_price() is None

    def test_time(self) -> None:
        ob = OrderBook(is_buy=True)
        assert ob.time == 0
        ob.set_time(time=10)
        assert ob.time == 10
        ob.update_time()
        assert ob.time == 11
