from typing import Dict
from typing import List

from .base import Logger
from .base import MarketStepEndLog


class MarketStepPrintLogger(Logger):
    """Logger of the market step class."""

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """Print the market log.

        Args:
            log (:class:`pams.logs.MarketStepEndLog`): matket log.

        Returns:
            None

        """
        print(
            f"{log.session.session_id} {log.market.get_time()} {log.market.market_id} "
            f"{log.market.name} {log.market.get_market_price()} "
            f"{log.market.get_fundamental_price()}"
        )


class MarketStepSaver(Logger):
    """Saver of the market step class."""

    def __init__(self) -> None:
        """Initialize the saver with an empty list of market step logs."""
        super().__init__()
        self.market_step_logs: List[Dict] = []

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """Stack the market log.

        Args:
            log (:class:`pams.logs.MarketStepEndLog`): market log.

        Returns:
            None

        """
        self.market_step_logs.append(
            {
                "session_id": log.session.session_id,
                "market_time": log.market.get_time(),
                "market_id": log.market.market_id,
                "market_name": log.market.name,
                "market_price": log.market.get_market_price(),
                "fundamental_price": log.market.get_fundamental_price(),
            }
        )
