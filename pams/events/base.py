import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import cast

from pams.market import Market


class EventHook:
    """Event Hook class.

    Event hook define when and what events are hooked from simulator.
    It means that Event ( :class:`Event` ) can be used for multiple time if you appropriately set the event hook using the evnet.
    Event can be hooked before and after order placements, order cancellations, order executions, each session, and each step mor markets.
    You can also filter hooking point by market time, classes, instances of markets. (Currently, only market class and instances are supported.)

    .. seealso:
        - :class:FundamentalPriceShock
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
            else:
                if hook_type == "market":
                    if specific_class is not None and not issubclass(
                        specific_class, Market
                    ):
                        raise ValueError("specific_class and hook_type is incompatible")
                    if specific_instance is not None and not isinstance(
                        specific_instance, Market
                    ):
                        raise ValueError(
                            "specific_instance and hook_type is incompatible"
                        )
                else:
                    raise AssertionError
        specific_class = cast(Optional[Type], specific_class)
        self.specific_class: Optional[Type] = specific_class
        self.specific_instance: Optional[object] = specific_instance

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | hook_type={self.hook_type}, "
            f"is_before={self.is_before}, time={self.time}, specific_class={self.specific_class}, "
            f"specific_instance={self.specific_instance}, event={self.event}>"
        )


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
        session: "Session",  # type: ignore  # NOQA
        simulator: "Simulator",  # type: ignore  # NOQA
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

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.event_id}, name={self.name}, "
            f"session={self.session}>"
        )

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
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

    def hooked_before_order(self, simulator: "Simulator", order: "Order") -> None:  # type: ignore  # NOQA
        """This method is hooked before order placements if you set the event hook.
        Please be careful that the order haven't yet been accepted by markets and it could be leakage of order information.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            order (Order): order accepting now.
        """
        pass

    def hooked_after_order(self, simulator: "Simulator", order_log: "OrderLog") -> None:  # type: ignore  # NOQA
        """This method is hooked after order placements if you set the event hook.
        Please be careful that the order haven't yet been executed if it could be executed immediately.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            order_log (OrderLog): order accepted.
        """
        pass

    def hooked_before_cancel(self, simulator: "Simulator", cancel: "Cancel") -> None:  # type: ignore  # NOQA
        """This method is hooked before order cancellations if you set the event hook.
        Please be careful that the cancel order haven't yet been executed.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            cancel (Cancel): cancel order submitted.
        """
        pass

    def hooked_after_cancel(self, simulator: "Simulator", cancel_log: "CancelLog") -> None:  # type: ignore  # NOQA
        """This method is hooked after order cancellations if you set the event hook.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            cancel_log (CancelLog): cancel order submitted.
        """
        pass

    def hooked_after_execution(self, simulator: "Simulator", execution_log: "ExecutionLog") -> None:  # type: ignore  # NOQA
        """This method is hooked after order executions if you set the event hook.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            execution_log (ExecutionLog): execution log.
        """
        pass

    def hooked_before_session(self, simulator: "Simulator", session: "Session") -> None:  # type: ignore  # NOQA
        """This method is hooked before session beginnings if you set the event hook.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            session (Session): session to be started.
        """
        pass

    def hooked_after_session(self, simulator: "Simulator", session: "Session") -> None:  # type: ignore  # NOQA
        """This method is hooked after session ends if you set the event hook.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            session (Session): session to be ended.
        """
        pass

    def hooked_before_step_for_market(self, simulator: "Simulator", market: "Market") -> None:  # type: ignore  # NOQA
        """This method is hooked at each step before market processing if you set the event hook.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            market (Market): market to be processed.
        """
        pass

    def hooked_after_step_for_market(self, simulator: "Simulator", market: "Market") -> None:  # type: ignore  # NOQA
        """This method is hooked at each step after market processing if you set the event hook.

        .. seealso:
            - ToDo: simulation flow

        Args:
            simulator (Simulator): simulator for reference.
            market (Market): market processed.
        """
        pass
