from typing import List
from typing import Union

from pams.agents.fcn_agent import FCNAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order


class MarketShareFCNAgent(FCNAgent):
    """Market Share FCN Agent class

    This agent submits orders based on market shares.
    This class inherits from the :class:`pams.agents.FCNAgent` class.
    """

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders based on FCN-based calculation and market shares.

        .. seealso::
            - :func:`pams.agents.FCNAgent.submit_orders`
        """
        filter_markets: List[Market] = [
            market
            for market in markets
            if self.is_market_accessible(market_id=market.market_id)
        ]
        if len(filter_markets) == 0:
            raise AssertionError("filter_markets in MarketShareFCNAgent is empty.")
        weights: List[float] = []
        for market in filter_markets:
            weights.append(float(self.get_sum_trade_volume(market=market)) + 1e-10)
        return super().submit_orders_by_market(
            market=self.get_prng().choices(filter_markets, weights=weights)[0]
        )

    def get_sum_trade_volume(self, market: Market) -> int:
        """get sum of trade volume.

        Args:
            market (:class:`pams.Market`): trading market.

        Returns:
            int: total trade volume.
        """
        t: int = market.get_time()
        t_start: int = max(0, t - self.time_window_size)
        volume: int = sum(market.get_executed_volumes(range(t_start, t + 1)))
        return volume
