import pytest

from pams import LIMIT_ORDER
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

    def test__repr__(self) -> None:
        ob = OrderBook(is_buy=True)
        assert str(ob) == "<pams.order_book.OrderBook | is_buy=True>"

    def test_time(self) -> None:
        ob = OrderBook(is_buy=True)
        assert ob.time == 0
        ob._set_time(time=10)
        assert ob.time == 10
        ob._update_time()
        assert ob.time == 11

    def test_get_price_volume(self) -> None:
        ob = OrderBook(is_buy=True)
        o = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        ob.add(o)
        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        ob.add(o)
        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.1
        )
        ob.add(o)
        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.1
        )
        ob.add(o)
        assert ob.get_price_volume() == {None: 1, 1.1: 2, 1.0: 1}
        ob = OrderBook(is_buy=False)
        o = Order(agent_id=0, market_id=0, is_buy=False, kind=MARKET_ORDER, volume=1)
        ob.add(o)
        o = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        ob.add(o)
        o = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.1
        )
        ob.add(o)
        o = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.1
        )
        ob.add(o)
        assert ob.get_price_volume() == {None: 1, 1.0: 1, 1.1: 2}

    def test_remove(self) -> None:
        ob = OrderBook(is_buy=False)
        o1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=MARKET_ORDER,
            volume=1,
            order_id=0,
        )
        ob.add(o1)
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        ob.add(o2)
        o3 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.1,
            order_id=2,
        )
        ob.add(o3)
        o4 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.1,
            order_id=3,
        )
        ob.add(o4)
        ob._remove(order=o2)
        o3.placed_at = None
        with pytest.raises(AssertionError):
            ob._remove(order=o3)

    def test_change_order_volume(self) -> None:
        ob = OrderBook(is_buy=False)
        o1 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=MARKET_ORDER,
            volume=1,
            order_id=0,
        )
        ob.add(o1)
        with pytest.raises(AssertionError):
            ob.change_order_volume(order=o1, delta=-2)
