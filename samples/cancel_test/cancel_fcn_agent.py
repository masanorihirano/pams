from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents import FCNAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order
from pams.utils.json_random import JsonRandom

# CANCEL_RATE of CancelFCNAgent in plhamJ
DEFAULT_CANCEL_RATE: float = 0.3


class CancelFCNAgent(FCNAgent):
    """Cancel FCN Agent class

    This agent submits orders in the same way as FCN agents and cancels each of them with the probability
    ``cancel_rate`` right after submitting it.
    This class inherits from the :class:`pams.agents.FCNAgent` class.
    """

    cancel_rate: float

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """agent setup.  Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. This can include the parameter "cancelRate"
                                       (0.3 if not specified) in addition to the parameters of
                                       :class:`pams.agents.FCNAgent`.
            accessible_markets_ids (List[int]): list of market IDs.
            *args: not used.
            **kwargs: not used.

        Returns:
            None
        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        if "cancelRate" in settings:
            json_random: JsonRandom = JsonRandom(prng=self.prng)
            self.cancel_rate = json_random.random(json_value=settings["cancelRate"])
        else:
            self.cancel_rate = DEFAULT_CANCEL_RATE
        if not 0.0 <= self.cancel_rate <= 1.0:
            raise ValueError("cancelRate have to be between 0.0 and 1.0")

    def submit_orders_by_market(self, market: Market) -> List[Union[Order, Cancel]]:
        """submit orders by market and cancel some of them (internal usage).

        Each order made by :func:`pams.agents.FCNAgent.submit_orders_by_market` is canceled with the probability
        ``cancel_rate``. The cancel orders follow all the orders, so that each order is placed before it is canceled.
        If an order is fully executed when it is placed, its cancel order has no effect.

        Args:
            market (Market): market to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        orders: List[Union[Order, Cancel]] = super().submit_orders_by_market(
            market=market
        )
        cancels: List[Union[Order, Cancel]] = []
        for order in orders:
            if isinstance(order, Order) and self.prng.random() < self.cancel_rate:
                cancels.append(Cancel(order=order))
        return orders + cancels
