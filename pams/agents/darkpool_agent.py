from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..market import Market
from ..order import Order, Cancel, MARKET_ORDER, LIMIT_ORDER
from .fcn_agent import FCNAgent

class DarkPoolFCNAgent(FCNAgent):
    """Dark Pool FCN Agent class

    This class inherits from the :class:`pams.agents.Agent` class.

    The agent perceives single pair of (litMarket, DarkPoolMarket)
    and decides which market to submit orders to with a given probability "d".
    """  # NOQA
    def setup(
        self, settings: Dict[str, Any],
        accessible_markets_ids: List[int]
    ) -> None:
        """agent setup.  Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration.
                                       This must include the parameters "d" and "DarkPoolMarket".
                                       This can include the parameters "fundamentalWeight", "chartWeight",
                                       "noiseWeight", "noiseScale", "timeWindowSize", "orderMargin", "marginType",
                                       and "meanReversionTime".
            accessible_markets_ids (List[int]): list of market IDs. len(accessible_markets_ids) must be 2.

        Returns:
            None
        """
        if "d" not in settings:
            raise ValueError("d is required for DarkPoolAgent")
        if "DarkPoolMarket" not in settings:
            raise ValueError("DarkPoolMarket is required for DarkPoolAgent")
        if len(accessible_markets_ids) != 2:
            raise ValueError("len(accessible_markets_ids) is not 2. Only single pair or litMarket and DarkPoolMarket can exists.")
        super().setup(settings=settings,
                    accessible_markets_ids=accessible_markets_ids)
        self.d: float = float(settings["d"])
        self.darkpool_market: Market = \
        self.simulator.name2market[settings["DarkPoolMarket"]]

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders based on FCN-based calculation.

        submit limit order to litMarket first, and then change the order destination to DarkPoolMarket with probability 1-d.
        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        markets.remove(self.darkpool_market)
        orders: List[Union[Order, Cancel]] = super().submit_orders(markets)
        markets.append(self.darkpool_market)
        is_order_to_lit: bool = \
        self.prng.choices([False,True], cum_weights=[1-self.d,1], k=1)[0]
        if not is_order_to_lit:
            for i, order in enumerate(orders):
                order.market_id = self.darkpool_market.market_id
                order.price = None
                order.kind = MARKET_ORDER
                orders[i] = order
        return orders