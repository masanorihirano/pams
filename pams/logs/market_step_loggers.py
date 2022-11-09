from typing import Dict
from typing import List

from .base import Logger
from .base import MarketStepEndLog


class MarketStepPrintLogger(Logger):
    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        print(
            f"{log.session.session_id} {log.market.get_time()} {log.market.market_id} {log.market.name} {log.market.get_market_price()} {log.market.get_fundamental_price()}"
        )


class MarketStepSaver(Logger):
    market_step_logs: List[Dict] = []

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
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
