import random
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..market import Market
from ..order import Order
from ..session import Session
from ..simulator import Simulator
from .base import EventABC
from .base import EventHook


class PriceLimitRule(EventABC):
    """This limits the price range.

    The order having the price that is out of the price range, the price is overridden to the edge of range.
    This event is only called via :func:`hooked_before_order` at designated step.
    """

    def __init__(
        self,
        event_id: int,
        prng: random.Random,
        session: Session,
        simulator: Simulator,
        name: str,
    ) -> None:
        super().__init__(
            event_id=event_id,
            prng=prng,
            session=session,
            simulator=simulator,
            name=name,
        )
        self.target_markets: Dict[str, Market] = {}
        self.is_enabled: bool = True
        self.activation_count: int = 0
        self.trigger_change_rate: float = 0.0

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "targetMarkets" and "triggerChangeRate".
                                       This can include the parameters "enabled".
                                       The parameter "referenceMarket" is obsoleted.

        Returns:
            None
        """
        if "referenceMarket" in settings:
            warnings.warn("referenceMarket is obsoleted")
        if "targetMarkets" not in settings:
            raise ValueError("targetMarkets is required for PriceLimitRule")
        if not isinstance(settings["targetMarkets"], list):
            raise ValueError("targetMarkets must be list")
        for market_name in settings["targetMarkets"]:
            if not isinstance(market_name, str):
                raise ValueError("constituent of targetMarkets have to be string")
            if market_name not in self.simulator.name2market:
                raise ValueError(f"{market_name} does not exist")
            market: Market = self.simulator.name2market[market_name]
            self.target_markets[market_name] = market
        if "triggerChangeRate" not in settings:
            raise ValueError("triggerChangeRate is required for PriceLimitRule")
        if not isinstance(settings["triggerChangeRate"], float):
            raise ValueError("triggerChangeRate have to be float.")
        self.trigger_change_rate = settings["triggerChangeRate"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = EventHook(
                event=self, hook_type="order", is_before=True, time=None
            )
            return [event_hook]
        else:
            return []

    def get_limited_price(self, order: Order, market: Market) -> Optional[float]:
        """Calculate the limited price for an order.

        Args:
            order (Order): order whose price is calculated
            market (Market): market that order belongs to

        Returns:
            Optional[float]: price after price limit. If the input order is market order, the return become None (market order).
        """
        reference_price = market.get_market_price(0)
        if market not in self.target_markets.values():
            raise AssertionError
        if order.price is None:
            return order.price
        order_price: float = order.price
        price_change: float = order_price - reference_price
        threshold_change: float = reference_price * self.trigger_change_rate
        if abs(price_change) >= abs(threshold_change):
            max_price: float = reference_price * (1 + self.trigger_change_rate)
            min_price: float = reference_price * (1 - self.trigger_change_rate)
            limited_price: float = min(max(order_price, min_price), max_price)
            return limited_price
        return order_price

    def hooked_before_order(self, simulator: Simulator, order: Order) -> None:
        new_price: Optional[float] = self.get_limited_price(
            order, simulator.id2market[order.market_id]
        )
        if order.price != new_price:
            self.activation_count += 1
        order.price = new_price


PriceLimitRule.hook_registration.__doc__ = EventABC.hook_registration.__doc__
PriceLimitRule.hooked_before_order.__doc__ = EventABC.hooked_before_order.__doc__
