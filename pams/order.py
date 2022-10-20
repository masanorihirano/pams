import warnings
from dataclasses import dataclass
from typing import Optional
from typing import cast


@dataclass(frozen=True)
class OrderKind:
    kind_id: int
    name: str

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if other.__class__ != self.__class__:
            return False
        other = cast(OrderKind, other)
        return other.kind_id == self.kind_id

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return self.kind_id


MARKET_ORDER = OrderKind(kind_id=0, name="MARKET_ORDER")
LIMIT_ORDER = OrderKind(kind_id=1, name="LIMIT_ORDER")


class Order:
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
        ttl: Optional[int] = 0,
    ):
        if kind == MARKET_ORDER and price is not None:
            raise ValueError("price have to be None when kind is MARKET_ORDER")
        if kind == LIMIT_ORDER and price is None:
            raise ValueError("price have to be set when kind is LIMIT_ORDER")
        if price is not None and price <= 0:
            warnings.warn("price should be positive")
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
        if agent_id != self.agent_id:
            raise AttributeError("agent_id is fake")
        if self.placed_at is not None:
            raise AttributeError(
                "this order is already submitted to a market or placed_at have to be set to None"
            )
        if self.is_canceled is True:
            raise AttributeError("this order is already canceled")

    def is_expired(self, time: int) -> bool:
        if self.placed_at is None:
            raise Exception("this order is not yet placed to a market")
        if self.ttl is None:
            return False
        else:
            return self.placed_at + self.ttl < time

    def extra_repr(self) -> str:
        return (
            f"id={self.order_id}, kind={self.kind}, is_buy={self.is_buy}, price={self.price}, volume={self.volume}, "
            + f"agent={self.agent_id}, market={self.market_id}, placed_at={self.placed_at}, ttl={self.ttl}, "
            + f"is_canceled={self.is_canceled}"
        )


class Cancel:
    def __init__(self, order: Order, placed_at: Optional[int] = None):
        self.order: Order = order
        self.placed_at: Optional[int] = placed_at

    def check_system_acceptable(self, agent_id: int) -> None:
        if agent_id != self.order.agent_id:
            raise AttributeError("canceling other's order")
        if self.placed_at is not None:
            raise AttributeError(
                "this cancel order is already submitted to a market or placed_at have to be set to None"
            )
        if self.order.is_canceled is True:
            raise AttributeError("this order is already canceled")
