import random
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

from .logs import Logger
from .market import Market


class IndexMarket(Market):
    """Index of market.

    This class inherits from the :class:`pams.market.Market` class.
    """

    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ):
        super().__init__(
            market_id=market_id,
            prng=prng,
            simulator=simulator,
            name=name,
            logger=logger,
        )
        self._components: List[Market] = []

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        """setup market configuration from setting format.

        Args:
            settings (Dict[str, Any]): market configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameter "markets".
                                       This should include the parameter "requires".

        Returns:
            None
        """
        super(IndexMarket, self).setup(settings, *args, **kwargs)
        if "markets" not in settings:
            raise ValueError("markets is required for index markets as components")
        if "requires" in settings:
            warnings.warn("requires in index market settings is no longer required")
        for market_name in settings["markets"]:
            market: Market = self.simulator.name2market[market_name]
            self._add_market(market=market)

    def _add_market(self, market: Market) -> None:
        """add market. (Internal method)

        Args:
            market (:class:`pams.market.Market`): market.

        Returns:
            None
        """
        if market in self._components:
            raise ValueError("market is already registered as components")
        if market.outstanding_shares is None:
            raise AssertionError(
                "outstandingShares is required in component market setting"
            )
        self._components.append(market)

    def _add_markets(self, markets: List[Market]) -> None:
        """add markets. (Internal method)

        Args:
            markets (List[:class:`pams.market.Market`]): list of market.

        Returns:
            None
        """
        for market in markets:
            self._add_market(market=market)

    def compute_fundamental_index(self, time: Optional[int] = None) -> float:
        """compute fundamental index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: fundamental index.

        .. note::
            In an index market, there are two types of fundamental values:
             1. fundamental value set from outside such as runner (:func:`pams.index_market.IndexMarket.compute_fundamental_index`)
             2. fundamental value calculated from components' fundamental value
                (:func:`pams.index_market.IndexMarket.get_fundamental_index`,
                 :func:`pams.index_market.IndexMarket.get_fundamental_price`)
            In usual usage, those become the same. But, in some special usage, it could be differ.
            This method return 1st one.
        """
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
        """compute market index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: market index.
        """
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
        """get components.

        Returns:
            List[:class:`pams.market.Market`]: list of components.
        """
        return self._components

    def is_all_markets_running(self) -> bool:
        """get whether all markets is running or not.

        Returns:
            bool: whether all markets is running or not.
        """
        return sum(map(lambda x: not x.is_running, self._components)) == 0

    def get_fundamental_index(self, time: Optional[int] = None) -> float:
        """get fundamental index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: fundamental index.

        .. note::
            In an index market, there are two types of fundamental values:
             1. fundamental value set from outside such as runner (:func:`pams.index_market.IndexMarket.compute_fundamental_index`)
             2. fundamental value calculated from components' fundamental value
                (:func:`pams.index_market.IndexMarket.get_fundamental_index`,
                 :func:`pams.index_market.IndexMarket.get_fundamental_price`)
            In usual usage, those become the same. But, in some special usage, it could be differ.
            This method return 2nd one.
        """
        return self.get_fundamental_price(time=time)

    def get_market_index(self, time: Optional[int] = None) -> float:
        """get computed market index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: market index.
        """
        return self.compute_market_index(time=time)

    def get_index(self, time: Optional[int] = None) -> float:
        """get market index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: market index.
        """
        return self.get_market_index(time=time)
