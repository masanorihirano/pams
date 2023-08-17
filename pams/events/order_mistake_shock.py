import random
import warnings
from typing import Any
from typing import Dict
from typing import List

from ..order import LIMIT_ORDER
from ..order import Order
from .base import EventABC
from .base import EventHook


class OrderMistakeShock(EventABC):
    """This suddenly changes the market price.

     It is as a consequence of a fat finger error, e.g., caused by a huge amount of orders
     at an extremely cheap or expensive price.

    This event is only called via :func:`hooked_before_step_for_market` at designated step.
    """

    target_market: "Market"  # type: ignore  # NOQA
    trigger_time: int
    price_change_rate: float
    order_volume: int

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
        self.order_time_length: int = 1
        self.triggerd: bool = False

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "target", "triggerTime", "priceChangeRate",
                                       "orderVolume", and "orderTimeLength". This can include the parameters "enabled".

        Returns:
            None
        """
        if "agent" in settings:
            warnings.warn("agent in OrderMistakeShock is obsoleted.")
        if "target" not in settings:
            raise ValueError("target is required for OrderMistakeShock")
        if settings["target"] not in self.simulator.name2market:
            raise ValueError(f"market {settings['target']} is not exists")
        self.target_market = self.simulator.name2market[settings["target"]]
        if "triggerTime" not in settings:
            raise ValueError("triggerTime is required for OrderMistakeShock")
        if not isinstance(settings["triggerTime"], int):
            raise ValueError("triggerTime have to be int")
        self.trigger_time = self.session.session_start_time + settings["triggerTime"]
        if "priceChangeRate" not in settings:
            raise ValueError("priceChangeRate is required for OrderMistakeShock")
        if not isinstance(settings["priceChangeRate"], float):
            raise ValueError("priceChangeRate have to be float")
        self.price_change_rate = settings["priceChangeRate"]
        if "orderVolume" not in settings:
            raise ValueError("orderVolume is required for OrderMistakeShock")
        if not isinstance(settings["orderVolume"], int):
            raise ValueError("orderVolume have to be int")
        self.order_volume = settings["orderVolume"]
        if "orderTimeLength" not in settings:
            raise ValueError("orderTimeLength is required for OrderMistakeShock")
        if not isinstance(settings["orderTimeLength"], int):
            raise ValueError("orderTimeLength have to be int")
        self.order_time_length = settings["orderTimeLength"]
        if "enabled" in settings:
            self.is_enabled = settings["enabled"]

    def hook_registration(self) -> List[EventHook]:
        if self.is_enabled:
            event_hook = EventHook(
                event=self, hook_type="order", is_before=True, time=[self.trigger_time]
            )
            return [event_hook]
        else:
            return []

    def hooked_before_order(self, simulator: "Simulator", order: "Order") -> None:  # type: ignore  # NOQA
        if not self.triggerd:
            market: "Market" = self.simulator.id2market[order.market_id]  # type: ignore  # NOQA
            base_price: float = market.get_market_price()
            order_price: float = base_price * (1 + self.price_change_rate)
            time_length: int = self.order_time_length
            # override a order
            order.is_buy = self.price_change_rate > 0.0
            order.kind = LIMIT_ORDER
            order.volume = self.order_volume
            order.price = order_price
            order.ttl = time_length
            self.triggerd = True


OrderMistakeShock.hook_registration.__doc__ = EventABC.hook_registration.__doc__
OrderMistakeShock.hooked_before_order.__doc__ = EventABC.hooked_before_order.__doc__
