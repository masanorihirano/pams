import random
import warnings
from typing import Any
from typing import Dict
from typing import List

from .base import EventABC
from .base import EventHook


class PriceLimitRule(EventABC):
    """This limits the price range.

    This event is only called via :func:`hooked_before_order` at designated step.
    """

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
        self.target_markets: Dict[str, "Market"] = {}  # type: ignore  # NOQA
        self.is_enabled: bool = True
        self.activation_count: int = 0

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "targetMarkets" and "triggerGhangeRate".
                                       This can include the parameters "enabled".
                                       The parameter "referenceMarket" is obsolate.

        Returns:
            None
        """
        if "referenceMarket" in settings:
            warnings.warn("referenceMarket is obsolate.")
        if "targetMarkets" not in settings:
            raise ValueError("targetMarkets is required for PriceLimitRule.")
        for market_name in settings["targetMarkets"]:
            market: "Market" = self.simulator.name2market[market_name]  # type: ignore  # NOQA
            self._add_market(name=market_name, market=market)
        if "triggerChangeRate" not in settings:
            raise ValueError("triggerChangeRate is required for PriceLimitRule.")
        if not isinstance(settings["triggerChangeRate"], float):
            raise ValueError("triggerChangeRate have to be float.")
        self.trigger_change_rate = settings["triggerChangeRate"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = []
            for instance in self.target_markets.values():
                event_hook.append(
                    EventHook(
                        event=self,
                        hook_type="market",
                        is_before=True,
                        time=None,
                        specific_instance=instance,
                    )
                )
            return event_hook
        else:
            return []

    def get_limited_price(self, order: "Order", market: "Market") -> float:  # type: ignore  # NOQA
        self.reference_price = market.get_market_price(0)
        if market not in self.target_markets.values():
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
        new_price: float = self.get_limited_price(order, simulator.id2market[order.market_id])  # type: ignore  # NOQA
        if order.price != new_price:
            self.activation_count += 1
        order.price = new_price

    def _add_market(self, name: str, market: "Market") -> None:  # type: ignore  # NOQA
        """add market. (Internal method)

        Args:
            market (:class:`pams.market.Market`): market.

        Returns:
            None
        """
        if name in self.target_markets:
            raise ValueError("market is already registered.")
        self.target_markets[name] = market


PriceLimitRule.hook_registration.__doc__ = EventABC.hook_registration.__doc__
PriceLimitRule.hooked_before_order.__doc__ = EventABC.hooked_before_order.__doc__
