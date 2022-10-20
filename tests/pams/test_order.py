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


# ToDo
