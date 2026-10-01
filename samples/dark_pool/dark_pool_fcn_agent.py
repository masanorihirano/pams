"""FCN agent of the dark pool sample."""

from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents import FCNAgent
from pams.market import Market
from pams.order import MARKET_ORDER
from pams.order import Cancel
from pams.order import Order
from pams.utils.json_random import JsonRandom
from samples.dark_pool.dark_pool_market import DarkPoolMarket


class DarkPoolFCNAgent(FCNAgent):
    """FCN agent that sends some of its orders to a dark pool.

    This class inherits from the :class:`pams.agents.FCNAgent` class.

    The agent orders only when it is asked for the orders to a
    :class:`samples.dark_pool.dark_pool_market.DarkPoolMarket`. It then decides the
    orders to the lit market of the dark pool in the same way as
    :class:`pams.agents.FCNAgent`, and sends each of them to the dark pool instead, as
    a market order, with the probability "darkPoolChance". So the agent has to be able
    to access both the dark pool and its lit market.
    """

    dark_pool_chance: float

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. In addition to the
                parameters of :class:`pams.agents.FCNAgent`, this must include the
                parameter "darkPoolChance", the probability to send an order to the
                dark pool.
            accessible_markets_ids (List[int]): list of market IDs.
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        if "darkPoolChance" not in settings:
            raise ValueError("darkPoolChance is required for DarkPoolFCNAgent")
        self.dark_pool_chance = JsonRandom(prng=self.prng).random(
            json_value=settings["darkPoolChance"]
        )
        if not 0.0 <= self.dark_pool_chance <= 1.0:
            raise ValueError("darkPoolChance must be between 0.0 and 1.0")

    def submit_orders_by_market(self, market: Market) -> List[Union[Order, Cancel]]:
        """Submit orders by market (internal usage).

        Args:
            market (Market): market to order. The agent orders only if it is an
                accessible dark pool.

        Returns:
            List[Union[Order, Cancel]]: order list.

        """
        if not isinstance(market, DarkPoolMarket):
            return []
        if not self.is_market_accessible(market_id=market.market_id):
            return []
        orders: List[Union[Order, Cancel]] = []
        for lit_order in super().submit_orders_by_market(market=market.lit_market):
            if (
                isinstance(lit_order, Order)
                and self.prng.random() < self.dark_pool_chance
            ):
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=lit_order.is_buy,
                        kind=MARKET_ORDER,
                        volume=lit_order.volume,
                        ttl=lit_order.ttl,
                    )
                )
            else:
                orders.append(lit_order)
        return orders
