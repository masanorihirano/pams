import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .logs import ExecutionLog
from .logs import Logger
from .logs import OrderLog
from .market import Market
from .order import Order
from .utils.json_random import JsonRandom


class Agent(ABC):
    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        self.agent_id: int = agent_id
        self.name: str = name
        self.asset_volumes: Dict[int, int] = {}
        self.cash_amount: float = 0
        self.prng: random.Random = prng
        self.sim: "Simulator" = simulator  # type: ignore
        self.logger: Optional[Logger] = logger

    def setup(self, settings: Dict[str, Any], accessible_markets_ids: List[int], *args, **kwargs) -> None:  # type: ignore
        if "cashAmount" not in settings:
            raise ValueError("cashAmount is required property of agent settings")
        self.cash_amount = settings["cashAmount"]
        if "assetVolume" not in settings:
            raise ValueError("cashAmount is required property of agent settings")

        for market_id in accessible_markets_ids:
            self.set_market_accessible(market_id=market_id)
            volume = int(
                JsonRandom(prng=self.prng).random(json_value=settings["assetVolume"])
            )
            self.set_asset_volume(market_id=market_id, volume=volume)

    def get_asset_volume(self, market_id: int) -> int:
        if not self.is_market_accessible(market_id=market_id):
            raise AssertionError(f"market {market_id} is not accessible")
        return self.asset_volumes[market_id]

    def get_cash_amount(self) -> float:
        return self.cash_amount

    def get_prng(self) -> random.Random:
        return self.prng

    def is_market_accessible(self, market_id: int) -> bool:
        return market_id in self.asset_volumes

    def submitted_order(self, log: OrderLog) -> None:
        pass

    def executed_order(self, log: ExecutionLog) -> None:
        pass

    def canceled_order(self, log: ExecutionLog) -> None:
        pass

    def set_asset_volume(self, market_id: int, volume: int) -> None:
        if not self.is_market_accessible(market_id=market_id):
            raise AssertionError(f"market {market_id} is not accessible")
        self.asset_volumes[market_id] = volume

    def set_cash_amount(self, cash_amount: float) -> None:
        self.cash_amount = cash_amount

    def set_market_accessible(self, market_id: int) -> None:
        self.asset_volumes[market_id] = 0

    @abstractmethod
    def submit_orders(self, markets: List[Market]) -> List[Order]:
        pass

    def update_asset_volume(self, market_id: int, delta: int) -> None:
        if not self.is_market_accessible(market_id=market_id):
            raise AssertionError(f"market {market_id} is not accessible")
        self.asset_volumes[market_id] += delta

    def update_cash_amount(self, delta: float) -> None:
        self.cash_amount += delta

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + str(id)
            + ","
            + str(self.cash_amount)
            + ","
            + str(self.asset_volumes)
        )
