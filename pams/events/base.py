from abc import ABC


class EventABC(ABC):
    def hook_by_time_and_simulator(self, simulator: "Simulator", time: int) -> None:  # type: ignore
        pass

    def hook_by_time_and_market(self, market: "Market", time: int) -> None:  # type: ignore
        pass

    def hook_by_time_and_agent(self, agent: "Agent", time: int) -> None:  # type: ignore
        pass
