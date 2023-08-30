from typing import Any
from typing import Dict
from typing import List
from typing import Optional
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

    This agent submits orders to the target market at the following price:
    :math:`\left\{\max_i(P^b_i) + \min_i(P^a_i) \pm P_f \times \theta\right\} / 2`
    where :math:`P^b_i` and :math:`P^a_i` are the best bid and ask prices of the :math:`i`-th target market,
    and :math:`P_f` is the fundamental price of the target market.
    """  # NOQA

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
        """agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration.
                                       This must include the parameters "targetMarket" and "netInterestSpread".
                                       This can include the parameters "orderTimeLength".
            accessible_markets_ids (List[int]): list of market IDs.

        Returns:
            None
        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        if "targetMarket" not in settings:
            raise ValueError("targetMarket is required for MarketMakerAgent.")
        if not isinstance(settings["targetMarket"], str):
            raise ValueError("targetMarket must be string")
        self.target_market = self.simulator.name2market[settings["targetMarket"]]
        if "netInterestSpread" not in settings:
            raise ValueError("netInterestSpread is required for MarketMakerAgent.")
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.net_interest_spread = json_random.random(
            json_value=settings["netInterestSpread"]
        )
        self.order_time_length = (
            int(json_random.random(json_value=settings["orderTimeLength"]))
            if "orderTimeLength" in settings
            else 2
        )

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders.

        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        orders: List[Union[Order, Cancel]] = []
        base_price: Optional[float] = self.get_base_price(markets=markets)
        if base_price is None:
            base_price = self.target_market.get_market_price()
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

    def get_base_price(self, markets: List[Market]) -> Optional[float]:
        """get base price of markets.

        Args:
            markets (List[:class:`pams.Market`]): markets.

        Returns:
            float, Optional: average of the max and min prices.
        """
        max_buy: float = -float("inf")
        for market in markets:
            best_buy_price: Optional[float] = market.get_best_buy_price()
            if (
                self.is_market_accessible(market_id=market.market_id)
                and best_buy_price is not None
            ):
                max_buy = max(max_buy, best_buy_price)
        min_sell: float = float("inf")
        for market in markets:
            best_sell_price: Optional[float] = market.get_best_sell_price()
            if (
                self.is_market_accessible(market_id=market.market_id)
                and best_sell_price is not None
            ):
                min_sell = min(min_sell, best_sell_price)
        if max_buy == -float("inf") or min_sell == float("inf"):
            return None
        return (max_buy + min_sell) / 2.0
