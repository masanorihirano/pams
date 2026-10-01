import math
import random
from typing import Any
from typing import Tuple

import pytest

from pams import LIMIT_ORDER
from pams import Market
from pams import Order
from pams import ProportionalTransactionFee
from pams import Simulator
from pams import TransactionFee
from pams.utils import find_class


def _make_market() -> Market:
    return Market(
        market_id=0,
        prng=random.Random(42),
        simulator=Simulator(prng=random.Random(42)),
        name="market",
    )


def _make_orders() -> Tuple[Order, Order]:
    buy_order = Order(
        agent_id=1, market_id=0, is_buy=True, kind=LIMIT_ORDER, volume=3, price=10.0
    )
    sell_order = Order(
        agent_id=0, market_id=0, is_buy=False, kind=LIMIT_ORDER, volume=3, price=10.0
    )
    return buy_order, sell_order


class ConstantTransactionFee(TransactionFee):
    def compute_fees(
        self, price: float, volume: int, buy_order: Order, sell_order: Order
    ) -> Tuple[float, float]:
        return 1.0, -0.5


class TestTransactionFee:
    def test_abstract(self) -> None:
        class NoComputeFees(TransactionFee):
            pass

        # instantiating these classes must fail, which is what is tested here
        # pylint: disable=abstract-class-instantiated
        with pytest.raises(TypeError):
            TransactionFee(market=_make_market())  # type: ignore
        with pytest.raises(TypeError):
            NoComputeFees(market=_make_market())  # type: ignore
        # pylint: enable=abstract-class-instantiated

    def test_init_and_setup(self) -> None:
        market = _make_market()
        fee = ConstantTransactionFee(market=market)
        assert fee.market is market
        # the default setup does nothing
        fee.setup(settings={"rate": 0.5, "unknown": [1, 2]})
        assert vars(fee) == {"market": market}
        buy_order, sell_order = _make_orders()
        assert fee.compute_fees(
            price=10.0, volume=3, buy_order=buy_order, sell_order=sell_order
        ) == (1.0, -0.5)

    def test_repr(self) -> None:
        market = _make_market()
        fee = ConstantTransactionFee(market=market)
        module = ConstantTransactionFee.__module__
        assert str(fee) == f"<{module}.ConstantTransactionFee | market={market}>"
        assert repr(ProportionalTransactionFee(market=market)) == (
            f"<pams.transaction_fees.ProportionalTransactionFee | market={market}>"
        )

    def test_find_class(self) -> None:
        assert find_class(name="TransactionFee") is TransactionFee
        assert (
            find_class(name="ProportionalTransactionFee") is ProportionalTransactionFee
        )


class TestProportionalTransactionFee:
    def test_init(self) -> None:
        market = _make_market()
        fee = ProportionalTransactionFee(market=market)
        assert isinstance(fee, TransactionFee)
        assert fee.market is market
        assert fee.rate == 0.0

    @pytest.mark.parametrize("rate", [0, 0.0, 0.001, 0.5, 0.999])
    def test_setup(self, rate: float) -> None:
        fee = ProportionalTransactionFee(market=_make_market())
        fee.setup(settings={"rate": rate, "unknown": "ignored"})
        assert fee.rate == rate
        assert isinstance(fee.rate, float)

    @pytest.mark.parametrize(
        "settings, match",
        [
            ({}, r"^rate is required for ProportionalTransactionFee$"),
            (
                {"makerRate": 0.001},
                r"^rate is required for ProportionalTransactionFee$",
            ),
            ({"rate": "0.001"}, r"^rate must be int or float$"),
            ({"rate": True}, r"^rate must be int or float$"),
            ({"rate": False}, r"^rate must be int or float$"),
            ({"rate": None}, r"^rate must be int or float$"),
            ({"rate": [0.001]}, r"^rate must be int or float$"),
            ({"rate": -0.001}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": -1}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": 1}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": 1.0}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": 10.0}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": math.nan}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": math.inf}, r"^rate must be in \[0.0, 1.0\)$"),
            ({"rate": -math.inf}, r"^rate must be in \[0.0, 1.0\)$"),
        ],
    )
    def test_setup_invalid(self, settings: Any, match: str) -> None:
        fee = ProportionalTransactionFee(market=_make_market())
        with pytest.raises(ValueError, match=match):
            fee.setup(settings=settings)
        assert fee.rate == 0.0

    @pytest.mark.parametrize(
        "rate, price, volume",
        [(0.0, 10.0, 3), (0.001, 10.0, 3), (0.002, 299.87, 7), (0.5, 0.1, 1)],
    )
    def test_compute_fees(self, rate: float, price: float, volume: int) -> None:
        fee = ProportionalTransactionFee(market=_make_market())
        fee.setup(settings={"rate": rate})
        buy_order, sell_order = _make_orders()
        expected = rate * price * volume
        assert fee.compute_fees(
            price=price, volume=volume, buy_order=buy_order, sell_order=sell_order
        ) == (expected, expected)
        # the orders are not changed
        assert buy_order.volume == 3
        assert sell_order.volume == 3
