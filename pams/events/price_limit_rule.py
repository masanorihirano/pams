import random
from typing import Any
from typing import Dict
from typing import List

from .base import EventABC
from .base import EventHook


class PriceLimitRule(EventABC):
    """This limits the price range.

    This event is only called via :func:`hooked_before_order` at designated step.
    """

    reference_market_name: str
    reference_market: "Market"  # type: ignore  # NOQA
    reference_price: float
    trigger_change_rate: float

    def __init__(
        self,
        event_id: int,
        prng: random.Random,
        session: "Session",  # type: ignore  # NOQA
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
    ) -> None:
        super().__init__(
            event_id=event_id,
            prng=prng,
            session=session,
            simulator=simulator,
            name=name,
        )
        self.is_enabled: bool = True

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "triggerDays", "target", "triggerTime", and "priceChangeRate".
                                       This can include the parameters "enabled" and "shockTimeLength".

        Returns:
            None
        """
        if "referenceMarket" not in settings:
            raise ValueError("referenceMarket is required for PriceLimitRule.")
        self.reference_market_name = settings["referenceMarket"]
        self.reference_market = self.simulator.name2market[self.reference_market_name]
        if "triggerChangeRate" not in settings:
            raise ValueError("triggerChangeRate is required for PriceLimitRule.")
        if not isinstance(settings["triggerChangeRate"], float):
            raise ValueError("triggerChangeRate have to be float.")
        self.trigger_change_rate = settings["triggerChangeRate"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = EventHook(
                event=self, hook_type="market", is_before=True, time=None
            )
            return [event_hook]
        else:
            return []

    def get_limited_price(self, order: "Order", market: "Market") -> float:  # type: ignore  # NOQA
        self.reference_price = self.reference_market.get_market_price(0)
        if self.reference_market != market:
            raise AssertionError
        order_price: float = order.price
        price_change: float = order_price - self.reference_price
        threshold_change: float = self.reference_price * self.trigger_change_rate
        if abs(price_change) >= abs(threshold_change):
            max_price: float = self.reference_price * (1 + self.trigger_change_rate)
            min_price: float = self.reference_price * (1 - self.trigger_change_rate)
            limited_price: float = min(max(order_price, min_price), max_price)
            return limited_price
        return order_price

    def hooked_before_order(self, simulator: "Simulator", order: "Order") -> None:  # type: ignore  # NOQA
        new_price: float = get_limited_price(order, self.reference_market)  # type: ignore  # NOQA
        order.price = new_price


PriceLimitRule.hook_registration.__doc__ = EventABC.hook_registration.__doc__
PriceLimitRule.hooked_before_order.__doc__ = EventABC.hooked_before_order.__doc__
