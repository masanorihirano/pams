import random
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents.high_frequency_agent import HighFrequencyAgent
from pams.market import Market
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
        self.net_interest_spread = json_random.random(json_value=settings["netInterestSpread"])
        self.order_time_length = json_random.random(json_value=settings["orderTimeLength"] if "orderTimeLength" in settings else "2")
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders based on FCN-based calculation.

        Create limit order to litMarket first, and then change the order destination to DarkPoolMarket with probability 1-d.
        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        filter_markets: List[Market] = self.filter_markets(markets)
        weights: List[float] = []
        for market in filter_markets:
            weights.append(float(self.get_sum_trade_volume(market)))
        if len(filter_markets) == 0:
            raise RuntimeError("filter_markets in MarketShareFCNAgent is empty.")
        return self.submit_orders_by_market(random.choice(filter_markets))

    def filter_markets(self, markets: List[Market]) -> List[Market]:
        a: List[Market] = []
        for market in markets:
            if self.is_market_accessible(market_id=market.market_id):
                a.append(market)
        return a

    def get_sum_trade_volume(self, market: Market) -> int:
        t: int = market.get_time()
        time_window_size: int = min(t, self.time_window_size)
        volume: int = 0
        for d in range(1, time_window_size + 1):
            volume += market.get_executed_volume(t - d)
        return volume
