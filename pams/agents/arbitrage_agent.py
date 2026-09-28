import random
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..index_market import IndexMarket
from ..logs import Logger
from ..market import Market
from ..order import LIMIT_ORDER
from ..order import Cancel
from ..order import Order
from .high_frequency_agent import HighFrequencyAgent


class ArbitrageAgent(HighFrequencyAgent):
    """Arbitrage Agent class.

    This class inherits from the HighFrequencyAgent class.
    Arbitrage agent mainly aimed to take arbitrage chance between spot markets and its index market.

    Note:
        Currently, index markets must have the same weight for each constitutional stock.

        Unlike plhamJ, the component markets of an index market are not made accessible automatically.
        Therefore, the accessible markets must include all the component markets of the index markets, too.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id, prng=prng, simulator=simulator, name=name, logger=logger
        )

        self.order_volume: int = 1
        self.order_threshold_price: float = 1.0
        self.order_time_length: int = 1

    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        """agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "orderVolume", "orderThresholdPrice".
                                       This can include the parameter "orderTimeLength".
            accessible_markets_ids (List[int]): list of market IDs.
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        Note:
            This warns if some component markets of an accessible index market are not accessible,
            or if no accessible market is an index market. Markets not registered to the simulator yet are ignored.
        """
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        if "orderVolume" not in settings:
            raise ValueError("orderVolume is required for ArbitrageAgent")
        if not isinstance(settings["orderVolume"], int):
            raise ValueError("orderVolume have to be int")
        self.order_volume = settings["orderVolume"]
        if "orderThresholdPrice" not in settings:
            raise ValueError("orderThresholdPrice is required for ArbitrageAgent")
        self.order_threshold_price = settings["orderThresholdPrice"]
        if "orderTimeLength" in settings:
            if not isinstance(settings["orderTimeLength"], int):
                raise ValueError("orderTimeLength have to be int")
            self.order_time_length = settings["orderTimeLength"]

        # runners set up all the markets before the agents, so the components of index markets are known here
        id2market: Dict[int, Market] = self.simulator.id2market
        accessible_markets: List[Market] = [
            id2market[market_id]
            for market_id in accessible_markets_ids
            if market_id in id2market
        ]
        index_markets: List[IndexMarket] = [
            market for market in accessible_markets if isinstance(market, IndexMarket)
        ]
        for index in index_markets:
            missing_markets: List[Market] = [
                market
                for market in index.get_components()
                if not self.is_market_accessible(market_id=market.market_id)
            ]
            if len(missing_markets) > 0:
                missing_market_names: str = ", ".join(
                    market.name for market in missing_markets
                )
                warnings.warn(
                    f"{self.__class__.__name__} {self.name} can access the index market {index.name} "
                    f"but not its component markets {missing_market_names}, so its orders to them will fail. "
                    "Add the groups of these markets to markets in the settings of this agent.",
                    stacklevel=2,
                )
        # a market not registered to the simulator yet may be an index market
        all_registered: bool = len(accessible_markets) == len(accessible_markets_ids)
        if len(index_markets) == 0 and all_registered:
            warnings.warn(
                f"{self.__class__.__name__} {self.name} cannot access any index market, so it never submits orders. "
                "Add an index market and its component markets to markets in the settings of this agent.",
                stacklevel=2,
            )

    def _submit_orders(self, market: Market) -> List[Union[Order, Cancel]]:
        """internal sub routine for submitting orders by market.

        Args:
            market (List[Market]): markets to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        orders: List[Union[Order, Cancel]] = []
        if not isinstance(market, IndexMarket):
            return orders
        if not self.is_market_accessible(market_id=market.market_id):
            return orders
        index: IndexMarket = market
        spots: List[Market] = index.get_components()
        if not index.is_running or not index.is_all_markets_running():
            return orders
        market_index: float = index.get_index()
        market_price: float = index.get_market_price()

        if len({x.outstanding_shares for x in spots}) > 1:
            raise NotImplementedError(
                "currently, the components must have the same outstanding shares"
            )

        if (
            market_price < market_index
            and market_index - market_price > self.order_threshold_price
        ):
            index_order_volume = len(spots) * self.order_volume
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=index.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=index_order_volume,
                    price=index.get_market_price(),
                    ttl=self.order_time_length,
                )
            )
            for m in spots:
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=m.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=self.order_volume,
                        price=m.get_market_price(),
                        ttl=self.order_time_length,
                    )
                )
        if (
            market_price > market_index
            and market_price - market_index > self.order_threshold_price
        ):
            index_order_volume = len(spots) * self.order_volume
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=index.market_id,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=index_order_volume,
                    price=index.get_market_price(),
                    ttl=self.order_time_length,
                )
            )
            for m in spots:
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=m.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=self.order_volume,
                        price=m.get_market_price(),
                        ttl=self.order_time_length,
                    )
                )
        return orders

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders to take arbitrage chance.

        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            orders.extend(self._submit_orders(market=market))
        return orders

    def __repr__(self) -> str:
        """string representation of FCN agent class.

        Returns:
            str: string representation of this class.
        """
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.agent_id}, rnd={self.prng}, "
            f"order_volume={self.order_volume}, order_threshold_price={self.order_threshold_price}, "
            f"order_time_length={self.order_time_length}>"
        )
