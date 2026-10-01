"""Tutorial: fundamentalists, chartists and noise traders (Chiarella and Iori, 2002)."""
import contextlib
import copy
import io
import random
import statistics
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence

import matplotlib.pyplot as plt

from pams.runners import SequentialRunner

SEED: int = 42
SEEDS: List[int] = list(range(42, 52))

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
                "highFrequencySubmitRate": 1.0,
            },
            {
                "sessionName": 1,
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
        "meanReversionTime": {"uniform": [50, 100]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
}
# [config-end]

# [cases-start]
CASES: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "more fundamental": {"fundamentalWeight": {"expon": [2.0]}},
    "more chart": {"chartWeight": {"expon": [0.5]}},
}
# [cases-end]


# [run-start]
def make_config(changes: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy the configuration and change the settings of the FCN agents.

    Args:
        changes (Mapping[str, Any]): keys of the ``FCNAgents`` block and their new
            values.

    Returns:
        Dict[str, Any]: the new configuration. ``CONFIG`` itself is not changed.

    """
    config: Dict[str, Any] = copy.deepcopy(CONFIG)
    config["FCNAgents"].update(copy.deepcopy(dict(changes)))
    return config


def run_simulation(seed: int, changes: Mapping[str, Any]) -> SequentialRunner:
    """Run the simulation with changed FCN agents.

    Args:
        seed (int): seed of the random number generator.
        changes (Mapping[str, Any]): keys of the ``FCNAgents`` block and their new
            values.

    Returns:
        SequentialRunner: the runner after the run.

    """
    runner = SequentialRunner(settings=make_config(changes), prng=random.Random(seed))
    # hide the two time lines that main() prints
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    return runner


# [run-end]


def deviation_stats(runner: SequentialRunner) -> Dict[str, float]:
    """Measure how the market price moves around the fundamental price.

    Args:
        runner (SequentialRunner): the runner after the run.

    Returns:
        Dict[str, float]: for the main session, the standard deviation (``std``) and
        the largest absolute value (``max``) of the market price minus the
        fundamental price, the number of times the market price crosses the
        fundamental price (``crosses``), and the number of shares traded
        (``volume``).

    """
    market = runner.simulator.name2market["Market"]
    main_session = runner.simulator.sessions[1]
    start = main_session.session_start_time
    times = range(start, start + main_session.iteration_steps)
    prices = market.get_market_prices(times=times)
    fundamentals = market.get_fundamental_prices(times=times)
    gaps: List[float] = [price - fp for price, fp in zip(prices, fundamentals)]
    return {
        "std": statistics.pstdev(gaps),
        "max": max(abs(gap) for gap in gaps),
        # a sign change of the gap between two steps is a cross
        "crosses": sum(1 for a, b in zip(gaps, gaps[1:]) if a * b < 0),
        "volume": sum(market.get_executed_volumes(times=times)),
    }


def compare(seeds: Sequence[int]) -> Dict[str, List[Dict[str, float]]]:
    """Run every case with every seed.

    Args:
        seeds (Sequence[int]): seeds of the runs.

    Returns:
        Dict[str, List[Dict[str, float]]]: for each case, the result of
        :func:`deviation_stats` for each seed.

    """
    return {
        name: [deviation_stats(run_simulation(seed, changes)) for seed in seeds]
        for name, changes in CASES.items()
    }


def print_table(title: str, rows: Mapping[str, Dict[str, float]]) -> None:
    """Print one line of results per case.

    Args:
        title (str): the line printed above the table.
        rows (Mapping[str, Dict[str, float]]): the results of each case.

    """
    print(title)
    print(f"{'case':<16} {'std':>6} {'max':>6} {'crosses':>8} {'volume':>7}")
    for name, row in rows.items():
        print(
            f"{name:<16} {row['std']:6.2f} {row['max']:6.2f} "
            f"{row['crosses']:8.1f} {row['volume']:7.1f}"
        )


def mean_rows(
    results: Mapping[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Average the results of each case over the seeds.

    Args:
        results (Mapping[str, List[Dict[str, float]]]): the result of
            :func:`compare`.

    Returns:
        Dict[str, Dict[str, float]]: the mean of each number for each case.

    """
    return {
        name: {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
        for name, rows in results.items()
    }


def plot_cases(seed: int, path: str) -> None:
    """Plot the market price of each case, one panel per case.

    Args:
        seed (int): seed of the runs.
        path (str): path of the image file to write.

    """
    fig, axes = plt.subplots(
        len(CASES), 1, sharex=True, sharey=True, figsize=(6.4, 7.2)
    )
    for ax, (name, changes) in zip(axes, CASES.items()):
        market = run_simulation(seed, changes).simulator.name2market["Market"]
        times = range(100, 600)
        ax.plot(times, market.get_market_prices(times=times), label="market price")
        ax.plot(
            times,
            market.get_fundamental_prices(times=times),
            color="black",
            label="fundamental price",
        )
        ax.set_title(name)
        ax.set_ylabel("price")
    axes[0].legend()
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    """Compare the cases with seed 42 and with ten seeds, and plot seed 42."""
    results = compare(SEEDS)
    print_table(
        f"seed {SEED}",
        {name: rows[SEEDS.index(SEED)] for name, rows in results.items()},
    )
    print_table(f"mean of seeds {SEEDS[0]} to {SEEDS[-1]}", mean_rows(results))
    plot_cases(SEED, path="ci2002_weights.png")


if __name__ == "__main__":
    main()
