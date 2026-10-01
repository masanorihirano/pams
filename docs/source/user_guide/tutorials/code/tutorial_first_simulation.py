"""Tutorial: run a simulation, then read and plot its results."""
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt

from pams.logs import MarketStepPrintLogger
from pams.logs import MarketStepSaver
from pams.runners import SequentialRunner

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
            },
        ],
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


def print_prices(seed: int) -> None:
    """Run the simulation and print the market price at every step.

    Args:
        seed (int): seed of the random number generator.

    """
    runner = SequentialRunner(
        settings=CONFIG, prng=random.Random(seed), logger=MarketStepPrintLogger()
    )
    runner.main()


def run_simulation(seed: int) -> Tuple[SequentialRunner, MarketStepSaver]:
    """Run the simulation and keep the prices of every step in memory.

    Args:
        seed (int): seed of the random number generator.

    Returns:
        Tuple[SequentialRunner, MarketStepSaver]: the runner after the run and the
        saver holding the prices.

    """
    saver = MarketStepSaver()
    runner = SequentialRunner(settings=CONFIG, prng=random.Random(seed), logger=saver)
    runner.main()
    return runner, saver


def summarize(runner: SequentialRunner) -> Dict[str, float]:
    """Read a few results of the main session from the simulator.

    Args:
        runner (SequentialRunner): the runner after ``main()`` has returned.

    Returns:
        Dict[str, float]: the last, lowest and highest market prices of the main
        session, the number of shares traded in it, and the number of shares held by
        all the agents at the end.

    """
    market = runner.simulator.name2market["Market"]
    main_session = runner.simulator.name2session["main"]
    start = main_session.session_start_time
    end = start + main_session.iteration_steps
    prices = market.get_market_prices(times=range(start, end))
    volumes = market.get_executed_volumes(times=range(start, end))
    agents = runner.simulator.agents
    return {
        "last_price": prices[-1],
        "min_price": min(prices),
        "max_price": max(prices),
        "total_volume": sum(volumes),
        "total_shares": sum(
            agent.get_asset_volume(market_id=market.market_id) for agent in agents
        ),
    }


def plot_prices(saver: MarketStepSaver, path: str) -> None:
    """Plot the market price and the fundamental price of the main session.

    Args:
        saver (MarketStepSaver): the saver used as the logger of the run.
        path (str): path of the image file to write.

    """
    logs: List[Dict[str, Any]] = [
        log for log in saver.market_step_logs if log["session_id"] == 1
    ]
    times = [log["market_time"] for log in logs]
    fig, ax = plt.subplots()
    ax.plot(times, [log["market_price"] for log in logs], label="market price")
    ax.plot(
        times, [log["fundamental_price"] for log in logs], label="fundamental price"
    )
    ax.set_xlabel("step")
    ax.set_ylabel("price")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Run all the steps of the tutorial with seed 42."""
    print_prices(seed=42)
    runner, saver = run_simulation(seed=42)
    print(summarize(runner))
    plot_prices(saver, path="prices.png")


if __name__ == "__main__":
    main()
