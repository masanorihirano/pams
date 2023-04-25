from typing import Any
from typing import Dict
from typing import List

from pams.logs.base import ExecutionLog
from pams.market import Market
from pams.order import Order


class DarkPoolMarket(Market):
    lit_market: Market

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """setup market configuration from setting format.

        Args:
            settings (Dict[str, Any]): market configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "tickSize", "litMarket" and
                                       either "marketPrice" or "fundamentalPrice".
                                       This can include the parameter "outstandingShares".

        Returns:
            None
        """
        super().setup(settings)
        if "litMarket" not in settings:
            raise ValueError("litMarket is required for DarkPoolMarket")
        self.lit_market: Market = self.simulator.name2market[settings["litMarket"]]

    def _execution(self) -> List[ExecutionLog]:
        """execute for market. (Usually, only triggered by runner)

        Returns:
            List[:class:`pams.logs.base.ExecutionLog`]: execution logs.
        """
        logs: List[ExecutionLog] = []
        while True:
            if (
                len(self.sell_order_book.priority_queue) == 0
                or len(self.buy_order_book.priority_queue) == 0
            ):
                break
            buy_order: Order = self.buy_order_book.priority_queue[0]
            sell_order: Order = self.sell_order_book.priority_queue[0]
            sell_buy_order_volumes: List[int] = [sell_order.volume, buy_order.volume]
            volume: int = min(sell_buy_order_volumes)
            execute_price: float
            if self.lit_market.get_mid_price() is None:
                execute_price = self.lit_market.get_market_price()
            else:
                execute_price = self.lit_market.get_mid_price()
            log = self._execute_orders(execute_price, volume, buy_order, sell_order)
            logs.append(log)
        return logs
