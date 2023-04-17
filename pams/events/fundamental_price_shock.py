import random
from typing import Any
from typing import Dict
from typing import List

from .base import EventABC
from .base import EventHook


class FundamentalPriceShock(EventABC):
    """This suddenly changes the fundamental price (just changing it).

    This event is only called via :func:`hooked_before_step_for_market` at designated step.
    """

    target_market_name: str
    target_market: "Market"  # type: ignore  # NOQA
    trigger_time: int
    price_change_rate: float

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
        self.shock_time_length: int = 1

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "triggerDays", "target", "triggerTime", and "priceChangeRate".
                                       This can include the parameters "enabled" and "shockTimeLength".

        Returns:
            None
        """
        if "triggerDays" in settings:
            raise ValueError("triggerDays and numStepsOneDay are obsoleted.")
        if "target" not in settings:
            raise ValueError("target is required for FundamentalPriceShock")
        self.target_market_name = settings["target"]
        if "triggerTime" not in settings:
            raise ValueError("triggerTime is required for FundamentalPriceShock")
        if not isinstance(settings["triggerTime"], int):
            raise ValueError("triggerTime have to be int")
        self.trigger_time = self.session.session_start_time + settings["triggerTime"]
        if "priceChangeRate" not in settings:
            raise ValueError("priceChangeRate is required for FundamentalPriceShock")
        self.price_change_rate = settings["priceChangeRate"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]
        if "shockTimeLength" in settings:
            if not isinstance(settings["shockTimeLength"], int):
                raise ValueError("shockTimeLength have to be int")
            self.shock_time_length = settings["shockTimeLength"]
        self.target_market = self.simulator.name2market[self.target_market_name]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = EventHook(
                event=self,
                hook_type="market",
                is_before=True,
                time=[self.trigger_time + i for i in range(self.shock_time_length)],
                specific_instance=self.target_market,
            )
            return [event_hook]
        else:
            return []

    def hooked_before_step_for_market(self, simulator: "Simulator", market: "Market") -> None:  # type: ignore  # NOQA
        time: int = market.get_time()
        if not (self.trigger_time <= time < self.trigger_time + self.shock_time_length):
            raise AssertionError
        if market != self.target_market:
            raise AssertionError
        market.change_fundamental_price(scale=1 + self.price_change_rate)


FundamentalPriceShock.hook_registration.__doc__ = EventABC.hook_registration.__doc__
FundamentalPriceShock.hooked_before_step_for_market.__doc__ = (
    EventABC.hooked_before_step_for_market.__doc__
)
