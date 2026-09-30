import math
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
    """Event that suddenly changes the market price.

    It is as a consequence of a fat finger error, e.g., caused by a huge amount of orders
    at an extremely cheap or expensive price.
    At the trigger time, the first order submitted to the target market is overridden by a limit order
    at the market price multiplied by (1 + priceChangeRate), with the volume orderVolume and the ttl orderTimeLength.
    It is a buy order if priceChangeRate is positive, otherwise a sell order.
    Like other orders, its price is rounded to the tick size by the market (down for buy, up for sell).
    This event is only called via :func:`hooked_before_order` at designated step.

    Note:
        The trigger time, price, side, volume, and ttl of the mistaken order follow OrderMistakeShock of plhamJ,
        but the way the order enters the market differs. plhamJ submits an additional order of the first agent
        at the beginning of the trigger step. This event creates no order: the agent whose order is overridden
        owns the mistaken order, which is processed like the original order (logs, executions, and callbacks).
        Therefore, nothing happens if no order is submitted to the target market at the trigger time.
        Also, while plhamJ silently skips a mistaken order with a negative price or a non-positive volume,
        :func:`setup` rejects a priceChangeRate of -1.0 or less and a non-positive orderVolume.
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
        """Initialize the order mistake shock (see :class:`pams.events.EventABC` for the arguments)."""
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
        """Event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "target", "triggerTime", "priceChangeRate",
                                       "orderVolume", and "orderTimeLength". This can include the parameters "enabled".

        Returns:
            None

        """
        if "agent" in settings:
            warnings.warn("agent in OrderMistakeShock is obsoleted.", stacklevel=2)
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
        if not math.isfinite(settings["priceChangeRate"]):
            raise ValueError("priceChangeRate have to be finite")
        if settings["priceChangeRate"] <= -1.0:
            raise ValueError("priceChangeRate have to be greater than -1.0")
        self.price_change_rate = settings["priceChangeRate"]
        if "orderVolume" not in settings:
            raise ValueError("orderVolume is required for OrderMistakeShock")
        if not isinstance(settings["orderVolume"], int):
            raise ValueError("orderVolume have to be int")
        if settings["orderVolume"] <= 0:
            raise ValueError("orderVolume have to be positive")
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
        return []

    def hooked_before_order(self, simulator: "Simulator", order: "Order") -> None:  # type: ignore  # NOQA
        if order.market_id != self.target_market.market_id:
            return
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
