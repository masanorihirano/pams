from typing import List
from typing import cast

from .market import Market


class IndexMarket(Market):
    components: List[Market] = []

    def _add_market(self, market: Market) -> None:
        if market in self.components:
            raise ValueError("market is already registered as components")
        if market.outstanding_shares is None:
            raise AssertionError(
                "outstandingShares is required in component market setting"
            )
        self.components.append(market)

    def _add_markets(self, markets: List[Market]) -> None:
        for market in markets:
            self._add_market(market=market)

    def compute_fundamental_index(self, time: int) -> float:
        total_value: float = 0
        total_shares: int = 0
        for market in self.components:
            total_value += (
                market.get_fundamental_price(time=time) * market.outstanding_shares
            )
            total_shares += cast(int, market.outstanding_shares)
        return total_value / total_shares

    def compute_market_index(self, time: int) -> float:
        total_value: float = 0
        total_shares: int = 0
        for market in self.components:
            total_value += (
                market.get_market_price(time=time) * market.outstanding_shares
            )
            total_shares += cast(int, market.outstanding_shares)
        return total_value / total_shares
