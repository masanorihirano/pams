from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents.fcn_agent import FCNAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order


class MarketShareFCNAgent(FCNAgent):
    """Market Share FCN Agent class

    This class inherits from the :class:`pams.agents.FCNAgent` class.
    """

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. See also :func:`pams.agents.FCNAgent.setup`.
            accessible_markets_ids (List[int]): list of market IDs. Length of accessible_markets_ids must be 2.

        Returns:
            None
        """
        if len(accessible_markets_ids) != 2:
            raise ValueError("length of accessible_markets_ids is not 2.")
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders based on FCN-based calculation.

        .. seealso::
            - :func:`pams.agents.FCNAgent.submit_orders`
        """
        filter_markets: List[Market] = self.filter_markets(markets=markets)
        if len(filter_markets) == 0:
            raise RuntimeError("filter_markets in MarketShareFCNAgent is empty.")
        weights: List[float] = []
        for market in filter_markets:
            weights.append(float(self.get_sum_trade_volume(market=market)))
        k: int = self.roulette(weights=weights)
        return self.submit_orders_by_market(market=self.get_prng().choices(filter_markets, weights=weights)[0])

    def filter_markets(self, markets: List[Market]) -> List[Market]:
        """filter markets by accessibility.

        Args:
            markets (List[:class:`pams.Market`]): markets before filtering.

        Returns:
            List[:class:`pams.Market`]:  markets after filtering.
        """
        a: List[Market] = []
        for market in markets:
            if self.is_market_accessible(market_id=market.market_id):
                a.append(market)
        return a

    def get_sum_trade_volume(self, market: Market) -> int:
        """get sum of trade volume.

        Args:
            market (:class:`pams.Market`): trading market.

        Returns:
            int: total trade volume.
        """
        t: int = market.get_time()
        time_window_size: int = min(t, self.time_window_size)
        volume: int = 0
        for d in range(1, time_window_size + 1):
            volume += market.get_executed_volume(t - d)
        return volume

    def roulette(self, weights: List[float]) -> int:
        """weighted roulette. Randomly get the index of the list.

        Args:
            weights (List[float]): rouleted list.

        Returns:
            int: index of the list.
        """
        size: int = len(weights)
        total: float = sum(weights)
        d: float = total * self.get_prng().random()
        w: float = 0.0
        for i in range(size):
            w += weights[i]
            if d <= w:
                return i
        return size - 1
