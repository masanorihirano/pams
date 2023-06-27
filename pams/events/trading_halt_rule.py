import random
import warnings
from typing import Any
from typing import Dict
from typing import List

from .base import EventABC
from .base import EventHook


class TradingHaltRule(EventABC):
    """A trading halt is a market regulation that suspends the trading of some assets.

    The current implementation sets :func:`pams.market.Market.is_running` = false when the price changed beyond the prespecified threshold range.
    """

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
        self.target_markets: Dict[str, "Market"] = {}  # type: ignore  # NOQA

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "targetMarkets" and "triggerChangeRate".
                                       This can include the parameters "haltingTimeLength" and "enabled".
                                       The parameter "referenceMarket" is obsolate.

        Returns:
            None
        """
        if "referenceMarket" in settings:
            warnings.warn("referenceMarket is obsolate.")
        if "targetMarkets" not in settings:
            raise ValueError("targetMarkets is required for TradingHaltRule")
        for market_name in settings["targetMarkets"]:
            market: "Market" = self.simulator.name2market[market_name]  # type: ignore  # NOQA
            self._add_market(name=market_name, market=market)
        if "triggerChangeRate" not in settings:
            raise ValueError("triggerChangeRate is required for TradingHaltRule.")
        if not isinstance(settings["triggerChangeRate"], float):
            raise ValueError("triggerChangeRate have to be float.")
        self.trigger_change_rate = settings["triggerChangeRate"]
        if "haltingTimeLength" in settings:
            if not isinstance(settings["haltingTimeLength"], int):
                raise ValueError("haltingTimeLength have to be int")
            self.halting_time_length = settings["haltingTimeLength"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = EventHook(event=self, hook_type="execution", is_before=False)
            return [event_hook]
        else:
            return []

    def hooked_after_execution(self, simulator: "Simulator", execution_log: "ExecutionLog") -> None:  # type: ignore  # NOQA
        market: "Market" = simulator.id2market[execution_log.market_id]  # type: ignore  # NOQA
        self.reference_price = market.get_market_price(0)
        if market.is_running:
            price_change: float = self.reference_price - market.get_market_price()
            threshold_change: float = (
                self.reference_price
                * self.trigger_change_rate
                * (self.activation_count + 1)
            )
            if abs(price_change) >= abs(threshold_change):
                for m in self.target_markets.values():
                    if m == market:
                        m._is_running = False
                self.halting_time_started = execution_log.time
                self.activation_count += 1

    def hooked_before_step_for_market(self, simulator: "Simulator", market: "Market") -> None:  # type: ignore  # NOQA
        if market.get_time() > self.halting_time_started + self.halting_time_length:
            for m in self.target_markets.values():
                if m == market:
                    m._is_running = True
            self.halting_time_started = 0

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


TradingHaltRule.hook_registration.__doc__ = EventABC.hook_registration.__doc__
TradingHaltRule.hooked_after_execution.__doc__ = EventABC.hooked_after_execution.__doc__
TradingHaltRule.hooked_before_step_for_market.__doc__ = (
    EventABC.hooked_before_step_for_market.__doc__
)
