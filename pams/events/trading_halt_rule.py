import random
import warnings
from typing import Any
from typing import Dict
from typing import List

from ..logs import ExecutionLog
from ..market import Market
from ..session import Session
from ..simulator import Simulator
from .base import EventABC
from .base import EventHook


class TradingHaltRule(EventABC):
    """A trading halt is a market regulation that suspends the trading of some assets.

    When the one of targetMarkets violate halting line, all of them will be halted.
    If you want to halt them separately, please make event separately.
    The current implementation sets :func:`pams.market.Market.is_running` = false when the price changed beyond the prespecified threshold range.
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
        self.is_enabled: bool = True
        self.halting_time_length: int = 1
        self.halting_time_started: int = 0
        self.activation_count: int = 0
        self.target_markets: Dict[str, Market] = {}
        self.trigger_change_rate: float = 0.0

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "targetMarkets" and "triggerChangeRate".
                                       This can include the parameters "haltingTimeLength" and "enabled".
                                       The parameter "referenceMarket" is obsoleted.

        Returns:
            None
        """
        if "referenceMarket" in settings:
            warnings.warn("referenceMarket is obsolete")
        if "targetMarkets" not in settings:
            raise ValueError("targetMarkets is required for TradingHaltRule")
        if not isinstance(settings["targetMarkets"], list):
            raise ValueError("targetMarkets must be list")
        for market_name in settings["targetMarkets"]:
            if not isinstance(market_name, str):
                raise ValueError("constituent of targetMarkets must be string")
            if market_name not in self.simulator.name2market:
                raise ValueError(f"{market_name} does not exist")
            market: Market = self.simulator.name2market[market_name]
            self.target_markets[market_name] = market
        if "triggerChangeRate" not in settings:
            raise ValueError("triggerChangeRate is required for TradingHaltRule")
        if not isinstance(settings["triggerChangeRate"], float):
            raise ValueError("triggerChangeRate have to be float")
        self.trigger_change_rate = settings["triggerChangeRate"]
        if "haltingTimeLength" not in settings:
            raise ValueError("haltingTimeLength is required")
        if not isinstance(settings["haltingTimeLength"], int):
            raise ValueError("haltingTimeLength have to be int")
        self.halting_time_length = settings["haltingTimeLength"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hooked_after_execution = EventHook(
                event=self, hook_type="execution", is_before=False
            )
            event_hooked_before_step_for_market = []
            for market in self.target_markets.values():
                event_hooked_before_step_for_market.append(
                    EventHook(
                        event=self,
                        hook_type="market",
                        is_before=True,
                        specific_instance=market,
                    )
                )
            return [event_hooked_after_execution] + event_hooked_before_step_for_market
        else:
            return []

    def hooked_after_execution(
        self, simulator: Simulator, execution_log: ExecutionLog
    ) -> None:
        """event to stop the trading."""
        market: Market = simulator.id2market[execution_log.market_id]
        reference_price = market.get_market_price(0)
        if market.is_running:
            price_change: float = reference_price - market.get_market_price()
            threshold_change: float = (
                reference_price * self.trigger_change_rate * (self.activation_count + 1)
            )
            if abs(price_change) >= abs(threshold_change):
                for m in self.target_markets.values():
                    if m == market:
                        m._is_running = False
                        self.halting_time_started = m.time
                        self.activation_count += 1
                        if simulator.current_session is None:
                            raise AssertionError
                        simulator.current_session.with_order_execution = False

    def hooked_before_step_for_market(
        self, simulator: Simulator, market: Market
    ) -> None:
        """event to start the trading."""
        # TODO: when halting is continued over session
        if market.get_time() > self.halting_time_started + self.halting_time_length:
            for m in self.target_markets.values():
                if m == market:
                    if simulator.current_session is None:
                        raise AssertionError
                    simulator.current_session.with_order_execution = True
                    m._is_running = True
                    self.halting_time_started = 0


TradingHaltRule.hook_registration.__doc__ = EventABC.hook_registration.__doc__
TradingHaltRule.hooked_after_execution.__doc__ = EventABC.hooked_after_execution.__doc__
TradingHaltRule.hooked_before_step_for_market.__doc__ = (
    EventABC.hooked_before_step_for_market.__doc__
)
