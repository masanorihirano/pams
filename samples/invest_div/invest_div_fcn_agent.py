"""Investment diversification FCN agent of the invest_div sample."""

import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pams.agents import FCNAgent
from pams.logs import Logger
from pams.market import Market
from pams.order import MARKET_ORDER
from pams.order import Cancel
from pams.order import Order


class InvestDivFCNAgent(FCNAgent):
    """Investment Diversification FCN Agent class.

    This agent submits orders in the same way as FCN agents under the regulation for investment diversification.
    This class inherits from the :class:`pams.agents.FCNAgent` class.

    When this agent has a chance to submit orders, it calculates its net asset value, i.e., the sum of its cash
    amount and the values of its assets in the accessible markets.

    - If the sum of the absolute asset values exceeds ``leverage_ratio`` times the net asset value, it submits
      no orders.
    - Otherwise, it makes the orders of :class:`pams.agents.FCNAgent`. The order for a market is kept as it is if
      the absolute asset value in the market is at most ``diversity_ratio`` times the net asset value. If not,
      it is replaced by a market order of one unit that reduces the position in the market. The market order is
      submitted only if it can be executed at once (see :meth:`is_market_order_executable`).

    Note:
        This is a port of ``InvestDivFCNAgent`` of plhamJ with two differences. In plhamJ, a local variable
        shadows the configured ``leverageRatio``, so the leverage limit is always 1.0, while this class uses the
        configured value. And plhamJ submits the market order even if it cannot be executed at once.

    References:
        - Nozaki, Mizuta, Yagi (2016) Investigation of the rule for investment diversification at the time of
          a market crash using an artificial market (in Japanese).
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the investment diversification FCN agent.

        See :class:`pams.agents.Agent` for the arguments.
        """
        super().__init__(
            agent_id=agent_id, prng=prng, simulator=simulator, name=name, logger=logger
        )
        self.leverage_ratio: float = 1.0
        self.diversity_ratio: float = 1.0

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. This must include the parameters "leverageRatio" and
                                       "diversityRatio" in addition to the parameters of
                                       :class:`pams.agents.FCNAgent`.
            accessible_markets_ids (List[int]): list of market IDs.
            *args: not used.
            **kwargs: not used.

        Returns:
            None
        """
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        if "leverageRatio" not in settings:
            raise ValueError("leverageRatio is required for InvestDivFCNAgent")
        if not isinstance(settings["leverageRatio"], (int, float)):
            raise ValueError("leverageRatio have to be float")
        self.leverage_ratio = float(settings["leverageRatio"])
        if "diversityRatio" not in settings:
            raise ValueError("diversityRatio is required for InvestDivFCNAgent")
        if not isinstance(settings["diversityRatio"], (int, float)):
            raise ValueError("diversityRatio have to be float")
        self.diversity_ratio = float(settings["diversityRatio"])

    def filter_markets(self, markets: List[Market]) -> List[Market]:
        """Filter the markets accessible by this agent.

        Args:
            markets (List[Market]): markets.

        Returns:
            List[Market]: accessible markets.
        """
        return [
            market
            for market in markets
            if self.is_market_accessible(market_id=market.market_id)
        ]

    def get_asset_value(self, market: Market) -> float:
        """Get the value of the asset held by this agent in the market.

        Args:
            market (Market): market.

        Returns:
            float: the asset volume times the market price. It is negative for a short position.
        """
        return market.get_market_price() * self.get_asset_volume(
            market_id=market.market_id
        )

    def is_market_order_executable(self, market: Market, is_buy: bool) -> bool:
        """Check whether a market order can be executed at once in the market.

        In plhamJ, a market order that cannot be executed at once stays in the order book with a NaN price, which
        no limit order matches, until it expires. In PAMS, it would be executed later against the next order on
        the other side at the price of that order. Therefore, this agent submits the market order of the
        regulation only if this method returns True.

        Args:
            market (Market): market.
            is_buy (bool): whether the market order is a buy order.

        Returns:
            bool: whether the market is running (i.e., orders are executed) and the order book on the other side
            has orders.

        """
        if not market.is_running:
            return False
        return len(market.sell_order_book if is_buy else market.buy_order_book) > 0

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """Submit orders based on FCN-based calculation under the regulation for investment diversification.

        Returns:
            List[Union[Order, Cancel]]: order list.

        .. seealso::
            - :func:`pams.agents.FCNAgent.submit_orders`
        """
        accessible_markets: List[Market] = self.filter_markets(markets=markets)
        net_asset_value: float = self.get_cash_amount()
        total_asset_value_abs: float = 0.0
        for market in accessible_markets:
            asset_value: float = self.get_asset_value(market=market)
            net_asset_value += asset_value
            total_asset_value_abs += abs(asset_value)

        orders: List[Union[Order, Cancel]] = []
        # plhamJ always uses 1.0 here because a local variable shadows the configured leverageRatio.
        if total_asset_value_abs > self.leverage_ratio * net_asset_value:
            return orders

        id2market: Dict[int, Market] = {
            market.market_id: market for market in accessible_markets
        }
        for order in super().submit_orders(markets=accessible_markets):
            market = id2market[order.market_id]
            if abs(self.get_asset_value(market=market)) <= (
                self.diversity_ratio * net_asset_value
            ):
                orders.append(order)
                continue
            # replace the order with a market order reducing the position by one unit (no optimization).
            asset_volume: int = self.get_asset_volume(market_id=market.market_id)
            if asset_volume != 0 and self.is_market_order_executable(
                market=market, is_buy=asset_volume < 0
            ):
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=asset_volume < 0,
                        kind=MARKET_ORDER,
                        volume=1,
                        ttl=10,  # the same as plhamJ (no effect if the order is executed at once)
                    )
                )
        return orders
