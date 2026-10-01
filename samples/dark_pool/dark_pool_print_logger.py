"""Print logger of the dark pool sample."""

import math
from typing import List
from typing import Optional
from typing import Tuple

from pams.logs.base import Logger
from pams.logs.base import MarketStepEndLog
from pams.market import Market
from samples.dark_pool.dark_pool_market import DarkPoolMarket


def find_market_pair(
    market: Market, markets: List[Market]
) -> Tuple[Market, Optional[DarkPoolMarket]]:
    """Find the lit market and the dark pool that a market belongs to.

    Args:
        market (Market): a lit market or a dark pool.
        markets (List[Market]): all the markets.

    Returns:
        Tuple[Market, Optional[DarkPoolMarket]]: the lit market and the dark pool. The
        dark pool is None if no dark pool has the market as its lit market.

    """
    if isinstance(market, DarkPoolMarket):
        return market.lit_market, market
    for dark_pool in markets:
        if isinstance(dark_pool, DarkPoolMarket) and dark_pool.lit_market is market:
            return market, dark_pool
    return market, None


def get_trade_price(lit_market: Market, dark_pool: Optional[DarkPoolMarket]) -> float:
    """Get the trade price of a lit market and its dark pool at the current step.

    Args:
        lit_market (Market): lit market.
        dark_pool (Optional[DarkPoolMarket]): dark pool of the lit market, or None if
            the lit market has no dark pool.

    Returns:
        float: the market price of the dark pool if the dark pool executed some orders
        at the current step, otherwise the market price of the lit market if the lit
        market executed some orders, and otherwise NaN.

    """
    if dark_pool is not None and dark_pool.get_executed_volume() > 0:
        return dark_pool.get_market_price()
    if lit_market.get_executed_volume() > 0:
        return lit_market.get_market_price()
    return math.nan


class DarkPoolPrintLogger(Logger):
    """Logger that prints the market steps of the dark pool sample.

    Each line shows one market at the end of one step, with the following columns
    separated by spaces: the session ID, the time, the market ID, the market name, the
    market price, the fundamental price, the trade price (see :func:`get_trade_price`)
    and the executed volume.
    """

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """Print the market log.

        Args:
            log (:class:`pams.logs.MarketStepEndLog`): market log.

        Returns:
            None

        """
        market: Market = log.market
        lit_market, dark_pool = find_market_pair(
            market=market, markets=log.simulator.markets
        )
        trade_price: float = get_trade_price(lit_market=lit_market, dark_pool=dark_pool)
        print(
            f"{log.session.session_id} {market.get_time()} {market.market_id} "
            f"{market.name} {market.get_market_price()} "
            f"{market.get_fundamental_price()} {trade_price} "
            f"{market.get_executed_volume()}"
        )
