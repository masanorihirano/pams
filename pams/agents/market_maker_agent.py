import random
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents.high_frequency_agent import HighFrequencyAgent
from pams.market import Market
from pams.order import LIMIT_ORDER
from pams.order import Cancel
from pams.order import Order
from pams.utils.json_random import JsonRandom


class MarketMakerAgent(HighFrequencyAgent):
    """Market Maker Agent class

    This class inherits from the :class:`pams.agents.Agent` class.
    """

    target_market: Market
    net_interest_spread: float
    order_time_length: int

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """agent setup.  Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration.
                                       This must include the parameters "targetMarket" and "netInterestSpread".
                                       This can include the parameters "orderTimeLength".
            accessible_markets_ids (List[int]): list of market IDs. Length of accessible_markets_ids must be 2.

        Returns:
            None
        """
        if "targetMarket" not in settings:
            raise ValueError("targetMarket is required for MarketMakerAgent.")
        self.target_market = self.simulator.name2market[settings["targetMarket"]]
        if "netInterestSpread" not in settings:
            raise ValueError("netInterestSpread is required for MarketMakerAgent.")
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.net_interest_spread = json_random.random(
            json_value=settings["netInterestSpread"]
        )
        self.order_time_length = json_random.random(
            json_value=settings["orderTimeLength"]
            if "orderTimeLength" in settings
            else "2"
        )
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders.

        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        base_price: float = self.get_base_price(markets)
        if base_price != float("inf"):
            base_price = self.target_market.get_market_price()
        orders: List[Order] = []
        price_margin: float = (
            self.target_market.get_fundamental_price() * self.net_interest_spread * 0.5
        )
        order_volume: int = 1
        orders.append(
            Order(
                agent_id=self.agent_id,
                market_id=self.target_market.market_id,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=order_volume,
                price=base_price - price_margin,
                ttl=self.order_time_length,
            )
        )
        orders.append(
            Order(
                agent_id=self.agent_id,
                market_id=self.target_market.market_id,
                is_buy=False,
                kind=LIMIT_ORDER,
                volume=order_volume,
                price=base_price + price_margin,
                ttl=self.order_time_length,
            )
        )
        return orders

    def get_base_price(self, markets: List[Market]) -> float:
        max_buy: float = -float("inf")
        for market in markets:
            if (
                self.is_market_accessible(market.market_id)
                and market.get_best_buy_price() is float
            ):
                max_buy = max(max_buy, market.get_best_buy_price())
        min_sell: float = float("inf")
        for market in markets:
            if (
                self.is_market_accessible(market.market_id)
                and market.get_best_sell_price() is float
            ):
                min_sell = min(min_sell, market.get_best_sell_price())
        return (max_buy + min_sell) / 2.0
