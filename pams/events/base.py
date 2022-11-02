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
        if hook_type not in ["order", "execution", "session", "market"]:
            raise ValueError(
                "hook type have to be order, execution, session, or market"
            )
        if hook_type == "execution" and is_before:
            raise ValueError("execution can be hooked only after it")
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
        name: str,
    ) -> None:
        self.event_id: int = event_id
        self.prng: random.Random = prng
        self.simulator = simulator
        self.name: str = name

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        pass

    @abstractmethod
    def hook_registration(self) -> List[EventHook]:
        pass

    def hooked_before_order(self, simulator: "Simulator", order: "Order") -> None:  # type: ignore
        pass

    def hooked_after_order(self, simulator: "Simulator", order_log: "OrderLog") -> None:  # type: ignore
        pass

    def hooked_after_execution(self, simulator: "Simulator", execution_log: "ExecutionLog") -> None:  # type: ignore
        pass

    def hooked_before_session(self, simulator: "Simulator", session: "Session") -> None:  # type: ignore
        pass

    def hooked_after_session(self, simulator: "Simulator", session: "Session") -> None:  # type: ignore
        pass

    def hooked_before_step_for_market(self, simulator: "Simulator", market: "Market") -> None:  # type: ignore
        pass

    def hooked_after_step_for_market(self, simulator: "Simulator", market: "Market") -> None:  # type: ignore
        pass
