import warnings
from dataclasses import dataclass
from typing import Optional
from typing import cast


@dataclass(frozen=True)
class OrderKind:
    """Kind of order.

    This class has an order kind ID and an order name.

    You should use the following pre-defined order kinds:
     - MARKET_ORDER
     - LIMIT_ORDER
    """

    kind_id: int
    name: str

    def __repr__(self) -> str:
        """String representation of this class.

        Returns:
            str: string representation of this class.

        """
        return self.name

    def __eq__(self, other: object) -> bool:
        """Get whether an argument's class is the same as this class or not.

        Args:
            other (object): an instance for comparison.

        Returns:
            bool: whether the class of the instance is the same as this class or not.

        """
        if other.__class__ != self.__class__:
            return False
        other = cast(OrderKind, other)
        return other.kind_id == self.kind_id

    def __ne__(self, other: object) -> bool:
        """Get whether an argument's class is different from this class or not.

        Args:
            other (object): an instance for comparison.

        Returns:
            bool: whether an argument's class is different from this class or not.

        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Get the order kind ID.

        Returns:
            int: order kind ID.

        """
        return self.kind_id


MARKET_ORDER = OrderKind(kind_id=0, name="MARKET_ORDER")
LIMIT_ORDER = OrderKind(kind_id=1, name="LIMIT_ORDER")


class Order:
    """Order class."""

    def __init__(
        self,
        agent_id: int,
        market_id: int,
        is_buy: bool,
        kind: OrderKind,
        volume: int,
        placed_at: Optional[int] = None,
        price: Optional[float] = None,
        order_id: Optional[int] = None,
        ttl: Optional[int] = None,
    ):
        """Initialization.

        Args:
            agent_id (int): agent ID.
            market_id (int): market ID.
            is_buy (bool): whether the order is buy order or not.
            kind (:class:`pams.order.OrderKind`): kind of order.
            volume (int): order volume.
            placed_at (int, Optional): time step that the order is placed.
                                       (Set by market. Please do not set it in agent)
            price (float, Optional): order price.
            order_id (int, Optional): order ID. (Set by market. Please do not set it in agent)
            ttl (int, Optional): time to order expiration.

        """
        if kind == MARKET_ORDER and price is not None:
            raise ValueError("price have to be None when kind is MARKET_ORDER")
        if kind == LIMIT_ORDER and price is None:
            raise ValueError("price have to be set when kind is LIMIT_ORDER")
        if price is not None and price <= 0:
            warnings.warn("price should be positive", stacklevel=2)
        if volume <= 0:
            raise ValueError("volume have to be positive")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl have to be positive or None")

        self.agent_id: int = agent_id
        self.market_id: int = market_id
        self.is_buy: bool = is_buy
        self.kind: OrderKind = kind
        self.volume: int = volume
        self.placed_at: Optional[int] = placed_at
        self.price: Optional[float] = price
        self.order_id: Optional[int] = order_id
        self.ttl: Optional[int] = ttl
        self.is_canceled: bool = False

    def check_system_acceptable(self, agent_id: int) -> None:
        """Check system acceptable (usually, markets automatically check it).

        Args:
            agent_id (int): agent ID.

        Returns:
            None

        """
        if agent_id != self.agent_id:
            raise AttributeError("agent_id is fake")
        if self.placed_at is not None:
            raise AttributeError(
                "this order is already submitted to a market or placed_at have to be set to None"
            )
        if self.is_canceled is True:
            raise AttributeError("this order is already canceled")

    def is_expired(self, time: int) -> bool:
        """Get whether the order is expired or not.

        Args:
            time (int): time to order expiration.

        Returns:
            bool: whether the order is expired or not.

        """
        if self.placed_at is None:
            raise ValueError("this order is not yet placed to a market")
        if self.ttl is None:
            return False
        return self.placed_at + self.ttl < time

    def __repr__(self) -> str:
        """Return the string representation of the order."""
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.order_id}, kind={self.kind}, "
            f"is_buy={self.is_buy}, price={self.price}, volume={self.volume}, "
            f"agent={self.agent_id}, market={self.market_id}, "
            f"placed_at={self.placed_at}, ttl={self.ttl}, is_canceled={self.is_canceled}>"
        )

    def _check_comparability(self, other: object) -> None:
        if self.__class__ != other.__class__:
            raise NotImplementedError(
                f"not supporting the comparison between Order and {other.__class__}"
            )
        other = cast(Order, other)
        if self.is_buy != other.is_buy:
            raise ValueError(
                "not supporting the comparison between sell and buy orders"
            )

    def __eq__(self, other: object) -> bool:
        """Return whether this order is equal to the other order.

        Two orders are equal when their order IDs, prices, placed time steps, sides
        (buy or sell) and kinds are the same. Only orders of the same side can be
        compared.
        """
        self._check_comparability(other)
        other = cast(Order, other)
        return (
            self.order_id == other.order_id
            and self.price == other.price
            and self.placed_at == other.placed_at
            and self.is_buy == other.is_buy
            and self.kind == other.kind
        )

    def _gt_lt(self, other: object, gt: bool = True) -> bool:
        # high priority is less
        self._check_comparability(other)
        other = cast(Order, other)

        def _compare_placed_at(a: Order, b: Order) -> bool:
            if a.placed_at is None and b.placed_at is None:
                raise ValueError("orders still not placed cannot be compared")
            if a.placed_at is None:
                return gt
            if b.placed_at is None:
                return not gt
            if a.placed_at != b.placed_at:
                return (
                    (a.placed_at > b.placed_at) if gt else (a.placed_at < b.placed_at)
                )
            if a.order_id is None or b.order_id is None:
                raise ValueError("orders still not placed cannot be compared")
            return (a.order_id > b.order_id) if gt else (a.order_id < b.order_id)

        if self.kind == MARKET_ORDER and other.kind == MARKET_ORDER:
            return _compare_placed_at(a=self, b=other)
        if self.kind == MARKET_ORDER:
            return not gt
        if other.kind == MARKET_ORDER:
            return gt
        if self.kind != LIMIT_ORDER or other.kind != LIMIT_ORDER:
            raise NotImplementedError
        if self.price is None or other.price is None:
            raise AssertionError
        if self.price != other.price:
            if self.is_buy:
                return (self.price < other.price) if gt else (self.price > other.price)
            return (self.price > other.price) if gt else (self.price < other.price)
        return _compare_placed_at(a=self, b=other)

    def __gt__(self, other: object) -> bool:
        """Return whether this order has a lower priority than the other order."""
        return self._gt_lt(other, gt=True)

    def __lt__(self, other: object) -> bool:
        """Return whether this order has a higher priority than the other order."""
        return self._gt_lt(other, gt=False)

    def __ne__(self, other: object) -> bool:
        """Return whether this order is not equal to the other order."""
        return not self.__eq__(other)

    def __le__(self, other: object) -> bool:
        """Return whether this order equals or has a higher priority than the other."""
        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other: object) -> bool:
        """Return whether this order equals or has a lower priority than the other."""
        return self.__eq__(other) or self.__gt__(other)


class Cancel:
    """Cancel order class."""

    def __init__(self, order: Order, placed_at: Optional[int] = None):
        """Initialization.

        Args:
            order (:class:`pams.order.Order`): order.
            placed_at (int, Optional): time step that the order is canceled.

        Returns:
            None

        """
        self.order: Order = order
        self.placed_at: Optional[int] = placed_at

    def __repr__(self) -> str:
        """Return the string representation of the cancel order."""
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | "
            f"placed_at={self.placed_at}, order={self.order}>"
        )

    @property
    def agent_id(self) -> int:
        """Getter for agent ID.

        Returns:
            int: agent ID.

        """
        return self.order.agent_id

    @property
    def market_id(self) -> int:
        """Getter for market ID.

        Returns:
            int: market ID.

        """
        return self.order.market_id

    def check_system_acceptable(self, agent_id: int) -> None:
        """Check system acceptable (usually, markets automatically check it).

        Args:
            agent_id (int): agent ID.

        Returns:
            None

        """
        if agent_id != self.order.agent_id:
            raise AttributeError("canceling other's order")
        if self.placed_at is not None:
            raise AttributeError(
                "this cancel order is already submitted to a market or placed_at have to be set to None"
            )
        if self.order.is_canceled is True:
            raise AttributeError("this order is already canceled")
