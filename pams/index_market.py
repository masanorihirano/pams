import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

from .market import Market


class IndexMarket(Market):
    _components: List[Market] = []

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        super(IndexMarket, self).setup(settings, *args, **kwargs)
        if "markets" not in settings:
            raise ValueError("markets is required for index markets as components")
        if "requires" in settings:
            warnings.warn("requires in index market settings is no longer required")
        for market_name in settings["markets"]:
            market: Market = self._simulator.name2market[market_name]
            self._add_market(market=market)

    def _add_market(self, market: Market) -> None:
        if market in self._components:
            raise ValueError("market is already registered as components")
        if market.outstanding_shares is None:
            raise AssertionError(
                "outstandingShares is required in component market setting"
            )
        self._components.append(market)

    def _add_markets(self, markets: List[Market]) -> None:
        for market in markets:
            self._add_market(market=market)

    def compute_fundamental_index(self, time: Optional[int] = None) -> float:
        if time is None:
            time = self.get_time()
        total_value: float = 0
        total_shares: int = 0
        for market in self._components:
            outstanding_shares = cast(int, market.outstanding_shares)
            total_value += market.get_fundamental_price(time=time) * outstanding_shares
            total_shares += cast(int, outstanding_shares)
        return total_value / total_shares

    def compute_market_index(self, time: Optional[int] = None) -> float:
        if time is None:
            time = self.get_time()
        total_value: float = 0
        total_shares: int = 0
        for market in self._components:
            outstanding_shares = cast(int, market.outstanding_shares)
            total_value += market.get_market_price(time=time) * outstanding_shares
            total_shares += outstanding_shares
        return total_value / total_shares

    def get_components(self) -> List[Market]:
        return self._components

    def is_all_markets_running(self) -> bool:
        return sum(map(lambda x: not x.is_running, self._components)) == 0

    def get_fundamental_index(self, time: Optional[int] = None) -> float:
        return self.get_fundamental_price(time=time)

    def get_market_index(self, time: Optional[int] = None) -> float:
        return self.compute_market_index(time=time)

    def get_index(self, time: Optional[int] = None) -> float:
        return self.get_market_index(time=time)
