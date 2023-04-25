import pytest

from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Cancel
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
        assert o != {}
        assert hash(o) == 0
        assert hash(o2) == 0
        assert hash(o3) == 1


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
        with pytest.warns(Warning):
            Order(
                agent_id=0,
                market_id=0,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=1,
                price=-10.1,
                ttl=1,
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

        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=1,
            order_id=1,
        )
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=MARKET_ORDER,
            volume=1,
            order_id=1,
        )
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
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
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
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        o.placed_at = 1
        o2.placed_at = 1
        assert (o > o2) is False
        assert (o < o2) is False
        o.price = 1.1
        assert o > o2
        assert o >= o2
        o.price = 0.9
        assert o < o2
        assert o <= o2
        o.price = 1.0
        o2.placed_at = 2
        assert o < o2
        assert o <= o2
        with pytest.raises(NotImplementedError):
            assert o == {}
        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=None,
            placed_at=1,
        )
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
            placed_at=1,
        )
        with pytest.raises(ValueError):
            assert o < o2
        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=4,  # type: ignore # NOQA
            volume=1,
            price=1.0,
            order_id=2,
            placed_at=1,
        )
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
            placed_at=1,
        )
        with pytest.raises(NotImplementedError):
            assert o < o2
        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=2,
            placed_at=1,
        )
        o.price = None
        o2 = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
            placed_at=1,
        )
        with pytest.raises(AssertionError):
            assert o < o2

    def test_ne__(self) -> None:
        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=2,
            placed_at=1,
        )
        assert not (o != o)
        assert o >= o
        assert o <= o

    def test__repr(self) -> None:
        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        assert (
            str(o)
            == "<pams.order.Order | id=1, kind=LIMIT_ORDER, is_buy=True, price=1.0, volume=1, agent=0, "
            "market=0, placed_at=None, ttl=None, is_canceled=False>"
        )


class TestCancel:
    def test__init__(self) -> None:
        o = Order(
            agent_id=0,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=1.0,
            order_id=1,
        )
        c = Cancel(order=o, placed_at=None)
        assert str(c) == f"<pams.order.Cancel | placed_at=None, order={o}>"
        assert c.agent_id == 0
        assert c.market_id == 0
        c.check_system_acceptable(agent_id=0)
        with pytest.raises(AttributeError):
            c.check_system_acceptable(agent_id=1)
        o.is_canceled = True
        with pytest.raises(AttributeError):
            c.check_system_acceptable(agent_id=0)
        c = Cancel(order=o, placed_at=10)
        with pytest.raises(AttributeError):
            c.check_system_acceptable(agent_id=0)
