"""Tutorial: write your own event and use it in a simulation."""
import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import cast

from pams import Market
from pams import Session
from pams import Simulator
from pams.events import EventABC
from pams.events import EventHook
from pams.runners import SequentialRunner


# [event-start]
class NewsShock(EventABC):
    """Random news that moves the fundamental price of one market at regular intervals.

    Every ``interval`` steps of its session, the fundamental price of the ``target``
    market is multiplied by :math:`e^{x}`, where :math:`x` is drawn from a normal
    distribution with mean 0 and standard deviation ``jumpScale``.
    """

    target_market: Market
    interval: int
    jump_scale: float
    news_times: List[int]

    def setup(self, settings: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Read the parameters of the event from its block of the config.

        Args:
            settings (Dict[str, Any]): the event's block of the config. It must include
                "target" (the name of a market), "interval" and "jumpScale".
            *args: not used.
            **kwargs: not used.

        """
        for key in ["target", "interval", "jumpScale"]:
            if key not in settings:
                raise ValueError(f"{key} is required for NewsShock")
        self.target_market = self.simulator.name2market[settings["target"]]
        self.interval = int(settings["interval"])
        self.jump_scale = float(settings["jumpScale"])
        self.news_times = []

    def hook_registration(self) -> List[EventHook]:
        """Tell the simulator when to call this event.

        Returns:
            List[EventHook]: a hook before the market step at every news time, and a
            hook after the session of the event.

        """
        start = self.session.session_start_time
        end = start + self.session.iteration_steps
        news_hook = EventHook(
            event=self,
            hook_type="market",
            is_before=True,
            time=list(range(start + self.interval, end, self.interval)),
            specific_instance=self.target_market,
        )
        session_hook = EventHook(
            event=self, hook_type="session", is_before=False, time=[end - 1]
        )
        return [news_hook, session_hook]

    def hooked_before_step_for_market(
        self, simulator: Simulator, market: Market
    ) -> None:
        """Change the fundamental price before the market step.

        Args:
            simulator (Simulator): the simulator.
            market (Market): the target market.

        """
        jump = self.prng.gauss(0.0, self.jump_scale)
        market.change_fundamental_price(scale=math.exp(jump))
        self.news_times.append(market.get_time())

    def hooked_after_session(self, simulator: Simulator, session: Session) -> None:
        """Print a summary at the end of the session of the event.

        Args:
            simulator (Simulator): the simulator.
            session (Session): the session that has just ended.

        """
        print(
            f"{self.name}: {len(self.news_times)} news in {session.name}, "
            f"fundamental price {self.target_market.get_fundamental_price():.2f}"
        )


# [event-end]

# [config-start]
CONFIG: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents"],
        "sessions": [
            {
                "sessionName": "warmup",
                "iterationSteps": 100,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
            },
            {
                "sessionName": "main",
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "events": ["News"],
            },
        ],
    },
    "News": {
        "class": "NewsShock",
        "target": "Market",
        "interval": 50,
        "jumpScale": 0.02,
    },
    "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents": 100,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
}
# [config-end]


# [run-start]
def run_simulation(seed: int) -> SequentialRunner:
    """Register the event class and run the simulation.

    Args:
        seed (int): seed of the random number generator.

    Returns:
        SequentialRunner: the runner after the run.

    """
    runner = SequentialRunner(settings=CONFIG, prng=random.Random(seed))
    runner.class_register(cls=NewsShock)
    runner.main()
    return runner


# [run-end]


def main() -> None:
    """Run the simulation with seed 42 and print the fundamental price at each news."""
    runner = run_simulation(seed=42)
    news = cast(NewsShock, runner.simulator.name2event["News"])
    market = runner.simulator.name2market["Market"]
    for time in news.news_times:
        before = market.get_fundamental_price(time=time - 1)
        after = market.get_fundamental_price(time=time)
        print(f"step {time}: {before:.2f} -> {after:.2f}")


if __name__ == "__main__":
    main()
