import pytest

from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Order
from pams.order import OrderKind


class TestOrderKind:
    def test_1(self) -> None:
        o = OrderKind(kind_id=0, name="test")
        assert o.kind_id == 0
        assert o.name == "test"
        assert str(o) == "test"
        o2 = OrderKind(kind_id=0, name="test2")
        o3 = OrderKind(kind_id=1, name="test3")
        assert o == o2
        assert o != o3


class TestOrder:
    def test_init(self) -> None:
        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10.1
        )
        o.check_system_acceptable(agent_id=0)
        with pytest.raises(AttributeError):
            o.check_system_acceptable(agent_id=1)
        with pytest.raises(ValueError):
            Order(agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1)
        with pytest.raises(ValueError):
            Order(
                agent_id=0,
                market_id=0,
                is_buy=True,
                kind=MARKET_ORDER,
                volume=1,
                price=10.1,
            )
        with pytest.raises(ValueError):
            Order(
                agent_id=0,
                market_id=0,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=-1,
                price=10.1,
            )
        with pytest.raises(ValueError):
            Order(
                agent_id=0,
                market_id=0,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=10.1,
                ttl=0,
            )
        o.placed_at = 1
        with pytest.raises(AttributeError):
            o.check_system_acceptable(agent_id=0)
        o.placed_at = None
        o.is_canceled = True
        with pytest.raises(AttributeError):
            o.check_system_acceptable(agent_id=0)

    def test_is_expired(self) -> None:
        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10.1
        )
        with pytest.raises(Exception):
            o.is_expired(time=10)
        o.placed_at = 0
        assert o.is_expired(time=10) is False
        o.ttl = 9
        assert o.is_expired(time=10) is True

    def test_gt(self) -> None:
        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10.1
        )
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=10.1,
        )
        with pytest.raises(ValueError):
            assert o > o2
        o2 = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=10.1
        )
        with pytest.raises(ValueError):
            assert o > o2
        o.placed_at = 1
        o2.placed_at = 2
        assert o < o2

        o = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        o2 = Order(agent_id=0, market_id=0, is_buy=True, kind=MARKET_ORDER, volume=1)
        o.placed_at = 1
        o2.placed_at = 1
        assert (o > o2) is False
        assert (o < o2) is False
        o2.placed_at = 2
        assert o < o2
        o2.placed_at = None
        assert o < o2
        o.placed_at = None
        o2.placed_at = 1
        assert o > o2

        o = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        o2 = Order(
            agent_id=0, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        o.placed_at = 1
        o2.placed_at = 1
        assert (o > o2) is False
        assert (o < o2) is False
        o.price = 1.1
        assert o < o2
        o.price = 0.9
        assert o > o2

        o = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        o2 = Order(
            agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=1, price=1.0
        )
        o.placed_at = 1
        o2.placed_at = 1
        assert (o > o2) is False
        assert (o < o2) is False
        o.price = 1.1
        assert o > o2
        o.price = 0.9
        assert o < o2
        o.price = 1.0
        o2.placed_at = 2
        assert o < o2
