import random
from typing import Any
from typing import Dict
from typing import List

from .base import EventABC
from .base import EventHook

from ..market import Market


class TradingHaltRule(EventABC):
    """A trading halt is a market regulation that suspends the trading of some assets.

    The current implementation sets :func:`pams.market.Market.is_running` = false when the price changed beyond the prespecified threshold range.
    """

    target_market_name: str
    target_markets: List[Market]  # type: ignore  # NOQA

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
        self.halting_time_length: int = 1
        self.halting_time_started: int = 0
        self.activation_count: int = 0

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
        self.reference_price = self.reference_market.get_market_price()
        if "triggerChangeRate" not in settings:
            raise ValueError("triggerChangeRate is required for TradingHaltRule.")
        if not isinstance(settings["triggerChangeRate"], float):
            raise ValueError("triggerChangeRate have to be float.")
        self.trigger_change_rate = settings["triggerChangeRate"]
        if "haltingTimeLength" in settings:
            if not isinstance(settings["haltingTimeLength"], int):
                raise ValueError("haltingTimeLength have to be int")
            self.halting_time_length = settings["haltingTimeLength"]
        if "targetMarkets" not in settings:
            raise ValueError("targetMarkets is required for TradingHaltRule")
        for market_name in settings["targetMarkets"]:
            self.target_markets.append(self.simulator.name2market[market_name])
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = EventHook(
                event=self,
                hook_type="market",
                is_before=True,
                time=[self.halting_time_started + i for i in range(self.halting_time_length)],
            )
            return [event_hook]
        else:
            return []

    def hooked_after_order(self, simulator: "Simulator", order_log: "OrderLog") -> None:  # type: ignore  # NOQA
        if self.reference_market.is_running():
            price_change: float = self.reference_price - self.reference_market.get_market_price()
            threshold_change: float = self.reference_price * self.trigger_change_rate * (self.activation_count + 1)
            if abs(price_change) >= abs(threshold_change):
                self.reference_market._is_running = False
                for m in self.target_market:
                    m._is_running = False
                self.halting_time_started = order_log.time
                self.activation_count += 1
        else:
            if order_log.time > self.halting_time_started + self.halting_time_length:
                self.reference_market._is_running = True
                for m in self.target_market:
                    m._is_running = True
                self.halting_time_started = 0


TradingHaltRule.hook_registration.__doc__ = EventABC.hook_registration.__doc__
TradingHaltRule.hooked_after_order.__doc__ = (
    EventABC.hooked_after_order.__doc__
)
