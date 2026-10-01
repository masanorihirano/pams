"""Tutorial: a price limit, and how long the price stays at it."""
import contextlib
import copy
import io
import random
import statistics
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import matplotlib.pyplot as plt

from pams.events import PriceLimitRule
from pams.runners import SequentialRunner

SEED: int = 42
SEEDS: Tuple[int, ...] = tuple(range(42, 52))
# the mean chart weights of the second experiment
CHART_WEIGHTS: Tuple[float, ...] = (0.0, 0.5, 1.0)
# a price closer to a limit than the tick size of the market is at the limit
TICK: float = 0.00001
# the cases of the first experiment: whether the limit is on, and the chart weight
CASES: Dict[str, Tuple[bool, float]] = {
    "limit": (True, 0.0),
    "no limit": (False, 0.0),
    "limit, chart 1.0": (True, 1.0),
}

# [config-start]
CONFIG: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents"],
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
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "events": ["PriceLimitRule"],
            },
        ],
    },
    "PriceLimitRule": {
        "class": "PriceLimitRule",
        "targetMarkets": ["Market"],
        "triggerChangeRate": 0.05,
        "enabled": True,
    },
    "Market": {
        "class": "Market",
        "tickSize": 0.00001,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
    },
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents": 100,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [0.2]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
}
# [config-end]


def run_simulation(
    seed: int, enabled: bool = True, chart_weight: float = 0.0, main_steps: int = 500
) -> SequentialRunner:
    """Run the simulation with or without the price limit.

    Args:
        seed (int): seed of the random number generator.
        enabled (bool): whether the price limit is on.
        chart_weight (float): mean of the chart weights of the FCN agents.
        main_steps (int): number of steps of the main session.

    Returns:
        SequentialRunner: the runner after the run.

    """
    config = copy.deepcopy(CONFIG)
    config["PriceLimitRule"]["enabled"] = enabled
    config["FCNAgents"]["chartWeight"] = {"expon": [chart_weight]}
    config["simulation"]["sessions"][1]["iterationSteps"] = main_steps
    runner = SequentialRunner(settings=config, prng=random.Random(seed))
    # main() prints the time taken, which is not needed here
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    return runner


def limit_stats(runner: SequentialRunner) -> Dict[str, float]:
    """Measure how long the price of the main session stays at the limits.

    Args:
        runner (SequentialRunner): the runner after the run.

    Returns:
        Dict[str, float]: the lowest and highest prices, the numbers of steps at or
        above the upper limit and at or below the lower limit, the longest run of
        consecutive steps at or beyond a limit, and the number of orders whose price
        the rule changed.

    """
    market = runner.simulator.name2market["Market"]
    main_session = runner.simulator.sessions[1]
    start = main_session.session_start_time
    prices: List[float] = market.get_market_prices(
        times=range(start, start + main_session.iteration_steps)
    )
    rate: float = CONFIG["PriceLimitRule"]["triggerChangeRate"]
    upper = market.get_market_price(0) * (1 + rate)
    lower = market.get_market_price(0) * (1 - rate)
    at_upper = [price >= upper - TICK for price in prices]
    at_lower = [price <= lower + TICK for price in prices]
    longest = 0
    current = 0
    for up, down in zip(at_upper, at_lower):
        current = current + 1 if up or down else 0
        longest = max(longest, current)
    # a disabled event registers no hooks and is not in name2event
    event = runner.simulator.name2event.get("PriceLimitRule")
    return {
        "lowest": min(prices),
        "highest": max(prices),
        "upper": sum(at_upper),
        "lower": sum(at_lower),
        "longest": longest,
        "changed": event.activation_count if isinstance(event, PriceLimitRule) else 0,
    }


def run_cases(seed: int) -> Dict[str, SequentialRunner]:
    """Run the three cases of the first experiment with the same seed.

    Args:
        seed (int): seed of the random number generator.

    Returns:
        Dict[str, SequentialRunner]: the runner of each case, by name.

    """
    return {
        name: run_simulation(seed, enabled=enabled, chart_weight=chart_weight)
        for name, (enabled, chart_weight) in CASES.items()
    }


def print_cases(runners: Dict[str, SequentialRunner]) -> None:
    """Print the statistics of each case.

    Args:
        runners (Dict[str, SequentialRunner]): the output of ``run_cases``.

    """
    print("case              lowest  highest  upper  lower  longest  changed")
    for name, runner in runners.items():
        stats = limit_stats(runner)
        print(
            f"{name:<16}  {stats['lowest']:6.2f}  {stats['highest']:7.2f}"
            f"  {stats['upper']:5d}  {stats['lower']:5d}  {stats['longest']:7d}"
            f"  {stats['changed']:7d}"
        )


def compare(
    seeds: Sequence[int] = SEEDS,
    chart_weights: Sequence[float] = CHART_WEIGHTS,
    main_steps: int = 500,
) -> Dict[float, List[Dict[str, float]]]:
    """Run the simulation with the limit for each chart weight and seed.

    Args:
        seeds (Sequence[int]): seeds of the runs.
        chart_weights (Sequence[float]): means of the chart weights.
        main_steps (int): number of steps of the main session.

    Returns:
        Dict[float, List[Dict[str, float]]]: the statistics of each run, by chart
        weight.

    """
    return {
        chart_weight: [
            limit_stats(
                run_simulation(seed, chart_weight=chart_weight, main_steps=main_steps)
            )
            for seed in seeds
        ]
        for chart_weight in chart_weights
    }


def print_comparison(results: Dict[float, List[Dict[str, float]]]) -> None:
    """Print the means over the seeds and the longest stay of each seed.

    Args:
        results (Dict[float, List[Dict[str, float]]]): the output of ``compare``.

    """
    print("chart weight  steps at a limit  longest stay  changed orders")
    for chart_weight, runs in results.items():
        at_limit = statistics.mean(run["upper"] + run["lower"] for run in runs)
        longest = statistics.mean(run["longest"] for run in runs)
        changed = statistics.mean(run["changed"] for run in runs)
        print(
            f"{chart_weight:12.1f}  {at_limit:16.1f}  {longest:12.1f}"
            f"  {changed:14.1f}"
        )
    print("longest stay of each seed:")
    for chart_weight, runs in results.items():
        stays = " ".join(f"{run['longest']:3d}" for run in runs)
        print(f"{chart_weight:12.1f}  {stays}")


def plot_prices(runners: Dict[str, SequentialRunner], path: str) -> None:
    """Plot the prices of the three cases.

    Args:
        runners (Dict[str, SequentialRunner]): the output of ``run_cases``.
        path (str): path of the image file to write.

    """
    times = range(100, 600)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6.4, 6.4))
    panels = [
        [("no limit", "tab:gray"), ("limit", "tab:red")],
        [("limit, chart 1.0", "tab:red")],
    ]
    for ax, cases in zip(axes, panels):
        for name, color in cases:
            market = runners[name].simulator.name2market["Market"]
            ax.plot(
                times, market.get_market_prices(times=times), color=color, label=name
            )
        ax.plot(
            times,
            market.get_fundamental_prices(times=times),
            color="black",
            label="fundamental price",
        )
        for limit in [285.0, 315.0]:
            ax.axhline(limit, color="black", linestyle="dashed", linewidth=0.8)
        ax.set_ylabel("price")
        ax.legend(loc="upper left")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    """Run the experiments of the tutorial and plot the prices of seed 42."""
    runners = run_cases(SEED)
    print_cases(runners)
    print_comparison(compare())
    plot_prices(runners, path="price_limit_prices.png")


if __name__ == "__main__":
    main()
