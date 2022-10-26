import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type


class EventHook:
    def __init__(
        self,
        event: "EventABC",
        hook_type: str,
        is_before: bool,
        time: Optional[List[int]] = None,
        specific_class: Optional[Type] = None,
    ):
        if hook_type not in ["simulator", "market", "agent"]:
            raise ValueError("hook type have to be simulator, market or agent")
        if hook_type == "simulator" and specific_class is not None:
            raise ValueError(
                "specific_class is not allowed to set if hook_type is simulator"
            )
        self.event = event
        self.hook_type = hook_type
        self.is_before = is_before
        self.time = time
        self.specific_class: Optional[Type] = (
            specific_class.__class__ if specific_class is not None else None
        )


class EventABC(ABC):
    def __init__(
        self,
        event_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore
        name: Optional[str] = None,
    ) -> None:
        self.event_id: int = event_id
        self.prng: random.Random = prng
        self.simulator = simulator
        self.name: Optional[str] = name

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        pass

    @abstractmethod
    def hook_registration(self) -> List[EventHook]:
        pass

    def hooked_by_time_and_simulator_before(self, simulator: "Simulator", time: int) -> None:  # type: ignore
        pass

    def hooked_by_time_and_simulator_after(self, simulator: "Simulator", time: int) -> None:  # type: ignore
        pass

    def hooked_by_time_and_market_before(self, market: "Market", time: int) -> None:  # type: ignore
        pass

    def hooked_by_time_and_market_after(self, market: "Market", time: int) -> None:  # type: ignore
        pass

    def hooked_by_time_and_agent_before(self, agent: "Agent", time: int) -> None:  # type: ignore
        pass

    def hooked_by_time_and_agent_after(self, agent: "Agent", time: int) -> None:  # type: ignore
        pass
