import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type


class EventHook:
    """Event Hook class.

    Event hook define when and what events are hooked from simulator.
    It means that Event ( :class:`Event` ) can be used for multiple time if you appropriately set the event hook using the evnet.
    Event can be hooked before and after order placements, order cancellations, order executions, each session, and each step mor markets.
    You can also filter hooking point by market time, classes, instances of markets. (Currently, only market class and instances are supported.)
    """

    def __init__(
        self,
        event: "EventABC",
        hook_type: str,
        is_before: bool,
        time: Optional[List[int]] = None,
        specific_class: Optional[Type] = None,
        specific_instance: Optional[object] = None,
    ):
        """Event Hook initialization.

        Args:
            event (:class:`pams.events.EventABC`): event instance.
            hook_type (str): hook type. This must be "order", "cancel", "execution", "session", or "market".
            is_before (bool): flag whether to run before or not. If hook_type is "execution", this must be set to False.
            time (List[int], Optional): event execution time.
            specific_class (Type, Optional): specific class. If this is specified, the hook_type must be "market".
                                             (In future, it could be expanded to other types.)
            specific_instance (object, Optional): specific instance. If this is specified, the hook_type must be "market".
                                                  (In future, it could be expanded to other types.)

        Returns:
            None
        """
        if hook_type not in ["order", "cancel", "execution", "session", "market"]:
            raise ValueError(
                "hook type have to be order, execution, session, or market"
            )
        if hook_type == "execution" and is_before:
            raise ValueError("execution can be hooked only after it")
        self.event = event
        self.hook_type = hook_type
        self.is_before = is_before
        self.time = time
        if specific_class is not None or specific_instance is not None:
            if hook_type not in ["market"]:
                raise ValueError(
                    "specific_class and specific_instance are not supported except for market"
                )
        self.specific_class: Optional[Type] = (
            specific_class.__class__ if specific_class is not None else None
        )
        self.specific_instance: Optional[object] = specific_instance


class EventABC(ABC):
    """event base class (ABC class).

    It defines what the event is.
    All events should inherit this class.

    .. seealso::
        - :class:`pams.events.FundamentalPriceShock`
    """

    def __init__(
        self,
        event_id: int,
        prng: random.Random,
        session: "Session",  # type: ignore
        simulator: "Simulator",  # type: ignore
        name: str,
    ) -> None:
        """event initialization. Usually be called from simulator/runner automatically.

        Args:
            event_id (int): event ID.
            prng (random.Random): pseudo random number generator for this event.
            session (:class:`pams.Session`): session to execute this event.
            simulator (:class:`pams.Simulator`): simulator that executes this event.
            name (str): event name.

        Returns:
            None
        """
        self.event_id: int = event_id
        self.prng: random.Random = prng
        self.simulator = simulator
        self.name: str = name
        self.session = session

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type:
        """event setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.

        Returns:
            None
        """
        pass

    @abstractmethod
    def hook_registration(self) -> List[EventHook]:
        """Define when this event should be hooked by simulator.
        This method is automatically hooked by simulator at the beginning of simulation.
        You must implement this.

        Returns:
            List[EventHook]: The list of event hook ( :class:`EventHook` )
        """
        pass

    def hooked_before_order(self, simulator: "Simulator", order: "Order") -> None:  # type: ignore
        pass

    def hooked_after_order(self, simulator: "Simulator", order_log: "OrderLog") -> None:  # type: ignore
        pass

    def hooked_before_cancel(self, simulator: "Simulator", cancel: "Cancel") -> None:  # type: ignore
        pass

    def hooked_after_cancel(self, simulator: "Simulator", cancel_log: "CancelLog") -> None:  # type: ignore
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
