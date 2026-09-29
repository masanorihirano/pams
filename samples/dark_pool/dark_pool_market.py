"""Dark pool market of the dark pool sample."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

from pams.logs.base import ExecutionLog
from pams.logs.base import OrderLog
from pams.market import Market
from pams.order import MARKET_ORDER
from pams.order import Cancel
from pams.order import Order


class DarkPoolMarket(Market):
    """Dark pool market class.

    This class inherits from the :class:`pams.market.Market` class.

    A dark pool does not show its order book. This market accepts only market orders
    and matches the buy and sell orders in time priority at the current mid price of
    its lit market, which is set by the parameter "markets". If one side of the order
    book of the lit market is empty, the market price of the lit market is used
    instead. The price is rounded to the tick size in favor of the earlier order: up
    when the buy order comes later and down when the sell order comes later.

    Orders are accepted only while this market is running. An order placed in a session
    without order execution is canceled as soon as it is placed.

    References:
        - Mizuta, T., Kosugi, S., Kusumoto, T., Matsumoto, W., & Izumi, K. (2015).
          Effects of dark pools on financial markets' efficiency and price discovery
          function: an investigation by multi-agent simulations. Evolutionary and
          Institutional Economics Review, 12(2), 375–394.
          https://doi.org/10.1007/s40844-015-0020-3

    """

    lit_market: Market

    def setup(self, settings: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Set up the market configuration from the setting format.

        Args:
            settings (Dict[str, Any]): market configuration. Usually, automatically set
                from json config of simulator. In addition to the parameters of
                :class:`pams.market.Market`, this must include the parameter
                "markets", a list with the name of the lit market.
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        """
        super().setup(settings, *args, **kwargs)
        if "markets" not in settings:
            raise ValueError("markets is required for DarkPoolMarket as the lit market")
        market_names: Any = settings["markets"]
        if not isinstance(market_names, list) or len(market_names) != 1:
            raise ValueError("markets of DarkPoolMarket must have exactly one market")
        if market_names[0] not in self.simulator.name2market:
            raise ValueError(f"lit market {market_names[0]} does not exist")
        lit_market: Market = self.simulator.name2market[market_names[0]]
        if isinstance(lit_market, DarkPoolMarket):
            raise ValueError("the lit market of DarkPoolMarket must not be a dark pool")
        self.lit_market = lit_market

    def get_lit_mid_price(self) -> float:
        """Get the current mid price of the lit market.

        Returns:
            float: the mean of the best buy and sell prices of the lit market, or the
            market price of the lit market if one side of its order book is empty.

        """
        best_buy_price: Optional[float] = self.lit_market.get_best_buy_price()
        best_sell_price: Optional[float] = self.lit_market.get_best_sell_price()
        if best_buy_price is None or best_sell_price is None:
            return self.lit_market.get_market_price()
        return (best_buy_price + best_sell_price) / 2.0

    def _add_order(self, order: Order) -> OrderLog:
        """Add order (usually, only triggered by runner).

        If this market is not running, the order is canceled right after it is placed.

        Args:
            order (:class:`pams.order.Order`): order. It must be a market order.

        Returns:
            :class:`pams.logs.base.OrderLog`: order log.

        """
        if order.kind != MARKET_ORDER:
            raise ValueError("DarkPoolMarket accepts only market orders")
        log: OrderLog = super()._add_order(order=order)
        if not self.is_running:
            self._cancel_order(cancel=Cancel(order=order))
        return log

    def remain_executable_orders(self) -> bool:
        """Check if there are remain executable orders in this market.

        All orders are market orders, so any buy order can be matched with any sell
        order.

        Returns:
            bool: whether some orders is executable or not.

        """
        return len(self.buy_order_book) > 0 and len(self.sell_order_book) > 0

    def _execution(self) -> List[ExecutionLog]:
        """Execute for market (usually, only triggered by runner).

        The best buy and sell orders are matched in time priority, and each pair is
        executed at the mid price of the lit market rounded in favor of the earlier
        order.

        Returns:
            List[:class:`pams.logs.base.ExecutionLog`]: execution logs.

        """
        logs: List[ExecutionLog] = []
        lit_mid_price: float = self.get_lit_mid_price()
        while self.remain_executable_orders():
            buy_order: Order = cast(Order, self.buy_order_book.get_best_order())
            sell_order: Order = cast(Order, self.sell_order_book.get_best_order())
            is_buy_later: bool = cast(int, buy_order.order_id) > cast(
                int, sell_order.order_id
            )
            # rounding as a sell price (up) favors the earlier sell order and rounding
            # as a buy price (down) favors the earlier buy order.
            price: float = self.convert_to_price(
                tick_level=self.convert_to_tick_level(
                    price=lit_mid_price, is_buy=not is_buy_later
                )
            )
            volume: int = min(buy_order.volume, sell_order.volume)
            logs.append(
                self._execute_orders(
                    price=price,
                    volume=volume,
                    buy_order=buy_order,
                    sell_order=sell_order,
                )
            )
        return logs
