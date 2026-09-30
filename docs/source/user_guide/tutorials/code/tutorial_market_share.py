"""Tutorial: two markets compete for the trading volume of one stock."""
import contextlib
import copy
import io
import random
import statistics
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import matplotlib.pyplot as plt

from pams import Market
from pams.runners import SequentialRunner

SEED: int = 42
SEEDS: List[int] = list(range(42, 52))
TICK_SIZES: List[float] = [10.0, 5.0, 2.0]
SPREADS: List[Optional[float]] = [None, 0.02, 0.01, 0.0001]


# [market-start]
class ExtendedMarket(Market):
    """A market whose traded volume at step 0 is set by the key ``tradeVolume``."""

    def setup(self, settings: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Set up the market, then set the volume of step 0.

        Args:
            settings (Dict[str, Any]): the market's block of the config. It can
                include "tradeVolume", an int.
            *args: passed to :meth:`pams.Market.setup`.
            **kwargs: passed to :meth:`pams.Market.setup`.

        """
        super().setup(settings, *args, **kwargs)
        if "tradeVolume" in settings:
            if not isinstance(settings["tradeVolume"], int):
                raise ValueError("tradeVolume must be int")
            self._executed_volumes = [int(settings["tradeVolume"])]


# [market-end]

# [config-start]
CONFIG_TICK: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market-A", "Market-B"],
        "agents": ["MarketShareFCNAgents"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": 100,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
            },
            {
                "sessionName": 1,
                "iterationSteps": 2000,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "maxHighFrequencyOrders": 1,
            },
        ],
    },
    "Market-A": {
        "class": "ExtendedMarket",
        "tickSize": 10.0,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
        "tradeVolume": 90,
    },
    "Market-B": {
        "class": "ExtendedMarket",
        "tickSize": 1.0,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
        "tradeVolume": 10,
    },
    "MarketShareFCNAgents": {
        "class": "MarketShareFCNAgent",
        "numAgents": 100,
        "markets": ["Market-A", "Market-B"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.2]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.0001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
        "marginType": "normal",
    },
}
# [config-end]

# [config-mm-start]
CONFIG_MM: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market-A", "Market-B"],
        "agents": ["MarketShareFCNAgents", "MarketMakerAgent"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": 100,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
            },
            {
                "sessionName": 1,
                "iterationSteps": 2000,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "maxHighFrequencyOrders": 1,
            },
        ],
    },
    "Market-A": {
        "class": "ExtendedMarket",
        "tickSize": 0.00001,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
        "tradeVolume": 90,
    },
    "Market-B": {
        "class": "ExtendedMarket",
        "tickSize": 0.00001,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
        "tradeVolume": 10,
    },
    "MarketShareFCNAgents": {
        "class": "MarketShareFCNAgent",
        "numAgents": 100,
        "markets": ["Market-A", "Market-B"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
    "MarketMakerAgent": {
        "class": "MarketMakerAgent",
        "numAgents": 1,
        "markets": ["Market-B"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "targetMarket": "Market-B",
        "netInterestSpread": 0.02,
        "orderTimeLength": 2,
    },
}
# [config-mm-end]


# [run-start]
def run_simulation(
    config: Dict[str, Any], seed: int, main_steps: Optional[int] = None
) -> SequentialRunner:
    """Run a simulation with the market class of this tutorial.

    Args:
        config (Dict[str, Any]): the configuration. It is not changed.
        seed (int): seed of the random number generator.
        main_steps (int, Optional): number of steps of the main session. If None,
            the number of the configuration is used.

    Returns:
        SequentialRunner: the runner after the run.

    """
    config = copy.deepcopy(config)
    if main_steps is not None:
        config["simulation"]["sessions"][1]["iterationSteps"] = main_steps
    runner = SequentialRunner(settings=config, prng=random.Random(seed))
    runner.class_register(cls=ExtendedMarket)
    # hide the two time lines printed by main()
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    return runner


def share_of_b(runner: SequentialRunner, window: int = 100) -> List[float]:
    """Compute the share of Market-B in the volume of each window of the main session.

    Args:
        runner (SequentialRunner): the runner after the run.
        window (int): the number of steps of a window.

    Returns:
        List[float]: the volume traded in Market-B divided by the volume traded in
        both markets, for each window. It is 0 when nothing is traded.

    """
    market_a: Market = runner.simulator.name2market["Market-A"]
    market_b: Market = runner.simulator.name2market["Market-B"]
    session = runner.simulator.sessions[1]
    start: int = session.session_start_time
    end: int = start + session.iteration_steps
    shares: List[float] = []
    for time in range(start, end, window):
        steps = range(time, time + window)
        volume_a: int = sum(market_a.get_executed_volumes(times=steps))
        volume_b: int = sum(market_b.get_executed_volumes(times=steps))
        shares.append(volume_b / max(volume_a + volume_b, 1))
    return shares


# [run-end]


def with_tick_size(tick_size: float) -> Dict[str, Any]:
    """Make a copy of ``CONFIG_TICK`` with another tick size in Market-A.

    Args:
        tick_size (float): the tick size of Market-A.

    Returns:
        Dict[str, Any]: the new configuration.

    """
    config: Dict[str, Any] = copy.deepcopy(CONFIG_TICK)
    config["Market-A"]["tickSize"] = tick_size
    return config


def with_spread(spread: Optional[float]) -> Dict[str, Any]:
    """Make a copy of ``CONFIG_MM`` with another spread of the market maker.

    Args:
        spread (float, Optional): the spread of the market maker. If None, the
            market maker is removed.

    Returns:
        Dict[str, Any]: the new configuration.

    """
    config: Dict[str, Any] = copy.deepcopy(CONFIG_MM)
    if spread is None:
        config["simulation"]["agents"].remove("MarketMakerAgent")
    else:
        config["MarketMakerAgent"]["netInterestSpread"] = spread
    return config


def print_windows(
    shares: Dict[str, List[float]], window: int = 100, start: int = 100
) -> None:
    """Print the share of Market-B in each window, one column per run.

    Args:
        shares (Dict[str, List[float]]): the shares of each run, by name.
        window (int): the number of steps of a window.
        start (int): the first step of the first window.

    """
    print("steps      " + "".join(f"{name:>11}" for name in shares))
    for i, row in enumerate(zip(*shares.values())):
        first: int = start + i * window
        steps: str = f"{first}-{first + window - 1}"
        print(f"{steps:<11}" + "".join(f"{share:11.2f}" for share in row))


# [compare-start]
def compare(
    configs: Dict[str, Dict[str, Any]], seeds: Sequence[int], main_steps: int
) -> Dict[str, List[List[float]]]:
    """Run each configuration with each seed.

    Args:
        configs (Dict[str, Dict[str, Any]]): the configurations, by name.
        seeds (Sequence[int]): the seeds.
        main_steps (int): number of steps of the main session.

    Returns:
        Dict[str, List[List[float]]]: for each configuration, and for each seed, the
        share of Market-B in each window of 100 steps.

    """
    return {
        name: [share_of_b(run_simulation(config, seed, main_steps)) for seed in seeds]
        for name, config in configs.items()
    }


# [compare-end]


def print_comparison(title: str, results: Dict[str, List[List[float]]]) -> None:
    """Print the share of Market-B in the last window of each run.

    Args:
        title (str): the title of the first column.
        results (Dict[str, List[List[float]]]): the results of :func:`compare`.

    """
    print(f"{title:>7}  B wins  mean share  share of B by seed")
    for name, runs in results.items():
        last: List[float] = [shares[-1] for shares in runs]
        wins: int = sum(1 for share in last if share > 0.5)
        print(
            f"{name:>7}  {wins:>2}/{len(last):<3}  {statistics.mean(last):10.2f}  "
            + " ".join(f"{share:.2f}" for share in last)
        )


def plot_tick(runners: Dict[str, SequentialRunner], path: str) -> None:
    """Plot the prices of the first run, and the share of Market-B in each run.

    Args:
        runners (Dict[str, SequentialRunner]): the runners after the run, by name.
        path (str): path of the image file to write.

    """
    fig, (ax_price, ax_share) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.4))
    first: SequentialRunner = next(iter(runners.values()))
    session = first.simulator.sessions[1]
    start: int = session.session_start_time
    times: List[int] = list(range(start, start + session.iteration_steps))
    for name, color in [("Market-A", "#2a78d6"), ("Market-B", "#eb6834")]:
        market: Market = first.simulator.name2market[name]
        ax_price.plot(
            times, market.get_market_prices(times=times), color=color, label=name
        )
    ax_price.plot(
        times,
        first.simulator.name2market["Market-A"].get_fundamental_prices(times=times),
        color="black",
        label="fundamental price",
    )
    ax_price.set_ylabel("price")
    ax_price.set_title(next(iter(runners)))
    ax_price.legend()
    # the runs get their own colors, not those of the markets above
    for (name, runner), color in zip(runners.items(), ["#4a3aa7", "#1baf7a"]):
        shares: List[float] = share_of_b(runner)
        ends: List[int] = [start + 100 * (i + 1) for i in range(len(shares))]
        ax_share.plot(ends, shares, color=color, marker="o", label=name)
    ax_share.set_ylim(-0.05, 1.05)
    ax_share.set_xlabel("step")
    ax_share.set_ylabel("share of B")
    ax_share.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_mean_shares(results: Dict[str, List[List[float]]], path: str) -> None:
    """Plot the share of Market-B in each window, averaged over the seeds.

    Args:
        results (Dict[str, List[List[float]]]): the results of :func:`compare`.
        path (str): path of the image file to write.

    """
    colors: List[str] = ["#9e9e9e", "#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots()
    for (name, runs), color in zip(results.items(), colors):
        means: List[float] = [statistics.mean(window) for window in zip(*runs)]
        ends: List[int] = [100 + 100 * (i + 1) for i in range(len(means))]
        ax.plot(ends, means, color=color, marker="o", label=f"spread {name}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("share of B, mean over the seeds")
    ax.legend()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    """Run all the steps of the tutorial."""
    runners: Dict[str, SequentialRunner] = {
        f"tick A {tick_size:g}": run_simulation(with_tick_size(tick_size), SEED)
        for tick_size in [10.0, 2.0]
    }
    print_windows({name: share_of_b(runner) for name, runner in runners.items()})
    plot_tick(runners, path="market_share_tick.png")
    ticks = compare(
        {f"{tick_size:g}": with_tick_size(tick_size) for tick_size in TICK_SIZES},
        seeds=SEEDS,
        main_steps=1000,
    )
    print_comparison("tick A", ticks)
    spreads = compare(
        {
            "none" if spread is None else f"{spread:g}": with_spread(spread)
            for spread in SPREADS
        },
        seeds=SEEDS,
        main_steps=500,
    )
    print_comparison("spread", spreads)
    plot_mean_shares(spreads, path="market_share_mm.png")


if __name__ == "__main__":
    main()
