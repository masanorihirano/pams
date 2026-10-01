"""Tutorial: a shock that spreads from one stock to another through arbitrage."""
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

from pams.runners import SequentialRunner

SEED: int = 42
SEEDS: List[int] = list(range(42, 52))

# [config-start]
CONFIG: Dict[str, Any] = {
    "simulation": {
        "markets": ["SpotMarket-1", "SpotMarket-2", "IndexMarket-I"],
        "agents": ["FCNAgents-1", "FCNAgents-2", "FCNAgents-I", "ArbitrageAgents"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": 100,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
                "maxNormalOrders": 3,
                "maxHighFrequencyOrders": 0,
            },
            {
                "sessionName": 1,
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
                "maxNormalOrders": 3,
                "maxHighFrequencyOrders": 5,
                "events": ["FundamentalPriceShock"],
            },
        ],
    },
    "FundamentalPriceShock": {
        "class": "FundamentalPriceShock",
        "target": "SpotMarket-1",
        "triggerTime": 0,
        "priceChangeRate": -0.1,
        "enabled": True,
    },
    "SpotMarket": {
        "class": "Market",
        "tickSize": 0.00001,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
    },
    "SpotMarket-1": {"extends": "SpotMarket"},
    "SpotMarket-2": {"extends": "SpotMarket"},
    "IndexMarket-I": {
        "class": "IndexMarket",
        "tickSize": 0.00001,
        "marketPrice": 300.0,
        "outstandingShares": 25000,
        "markets": ["SpotMarket-1", "SpotMarket-2"],
    },
    "FCNAgent": {
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
    "FCNAgents-1": {"extends": "FCNAgent", "markets": ["SpotMarket-1"]},
    "FCNAgents-2": {"extends": "FCNAgent", "markets": ["SpotMarket-2"]},
    "FCNAgents-I": {"extends": "FCNAgent", "markets": ["IndexMarket-I"]},
    "ArbitrageAgents": {
        "class": "ArbitrageAgent",
        "numAgents": 100,
        "markets": ["IndexMarket-I", "SpotMarket-1", "SpotMarket-2"],
        "assetVolume": 50,
        "cashAmount": 150000,
        "orderVolume": 1,
        "orderThresholdPrice": 1.0,
    },
}
# [config-end]

# [cases-start]
# the mean fundamental weight of the FCN agents of spot 1 and of the index
CASES: Dict[str, Tuple[float, float]] = {
    "same speed": (1.0, 1.0),
    "spot reacts faster": (1.0, 0.1),
    "index reacts faster": (0.1, 1.0),
}
# [cases-end]


# [run-start]
def run_simulation(
    seed: int,
    spot_weight: float = 1.0,
    index_weight: float = 1.0,
    main_steps: int = 500,
) -> SequentialRunner:
    """Run the simulation with the given fundamental weights.

    Args:
        seed (int): seed of the random number generator.
        spot_weight (float): mean fundamental weight of the FCN agents of spot 1.
        index_weight (float): mean fundamental weight of the FCN agents of the index.
        main_steps (int): number of steps of the main session.

    Returns:
        SequentialRunner: the runner after the run.

    """
    config: Dict[str, Any] = copy.deepcopy(CONFIG)
    config["FCNAgents-1"]["fundamentalWeight"] = {"expon": [spot_weight]}
    config["FCNAgents-I"]["fundamentalWeight"] = {"expon": [index_weight]}
    config["simulation"]["sessions"][1]["iterationSteps"] = main_steps
    runner = SequentialRunner(settings=config, prng=random.Random(seed))
    # hide the two time lines that main() prints
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    return runner


# [run-end]


def print_prices(runner: SequentialRunner, times: Sequence[int]) -> None:
    """Print the market prices of the three markets and two fundamental prices.

    Args:
        runner (SequentialRunner): the runner after the run.
        times (Sequence[int]): the steps to print.

    """
    spot1 = runner.simulator.name2market["SpotMarket-1"]
    spot2 = runner.simulator.name2market["SpotMarket-2"]
    index = runner.simulator.name2market["IndexMarket-I"]
    print(
        f"{'step':>4} {'spot 1':>7} {'spot 2':>7} {'index':>7} "
        f"{'fund 1':>7} {'fund I':>7}"
    )
    for t in times:
        print(
            f"{t:4d} {spot1.get_market_price(t):7.2f} {spot2.get_market_price(t):7.2f} "
            f"{index.get_market_price(t):7.2f} {spot1.get_fundamental_price(t):7.2f} "
            f"{index.get_fundamental_price(t):7.2f}"
        )


def plot_prices(runner: SequentialRunner, path: str) -> None:
    """Plot the market prices and the fundamental prices of the three markets.

    Args:
        runner (SequentialRunner): the runner after the run.
        path (str): path of the image file to write.

    """
    main_session = runner.simulator.sessions[1]
    start = main_session.session_start_time
    times = range(start - 20, start + main_session.iteration_steps)
    fig, ax = plt.subplots()
    for name, label, color in [
        ("SpotMarket-1", "spot 1", "tab:red"),
        ("SpotMarket-2", "spot 2", "tab:green"),
        ("IndexMarket-I", "index", "tab:blue"),
    ]:
        market = runner.simulator.name2market[name]
        ax.plot(times, market.get_market_prices(times=times), color=color, label=label)
        ax.plot(
            times,
            market.get_fundamental_prices(times=times),
            color="black",
            linestyle="--",
            linewidth=1,
            # one legend entry for the three fundamental prices
            label="fundamental" if name == "IndexMarket-I" else None,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("price")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def measure(runner: SequentialRunner) -> Dict[str, float]:
    """Measure the reaction of spot 2 and of the arbitrage agents in the main session.

    Args:
        runner (SequentialRunner): the runner after the run.

    Returns:
        Dict[str, float]: the mean gap between the index market price and the
        index value (``premium``), the mean market price of spot 2 minus its
        fundamental price 300 (``spot 2``), and the net number of shares of spot 2
        and of the index bought by all the arbitrage agents (``arb spot 2`` and
        ``arb index``).

    """
    simulator = runner.simulator
    spot1 = simulator.name2market["SpotMarket-1"]
    spot2 = simulator.name2market["SpotMarket-2"]
    index = simulator.name2market["IndexMarket-I"]
    main_session = simulator.sessions[1]
    start = main_session.session_start_time
    times = range(start, start + main_session.iteration_steps)
    spot1_prices = spot1.get_market_prices(times=times)
    spot2_prices = spot2.get_market_prices(times=times)
    index_prices = index.get_market_prices(times=times)
    arbitrage_agents = simulator.agents_group_name2agent["ArbitrageAgents"]
    initial: int = 50 * len(arbitrage_agents)
    return {
        "premium": statistics.mean(
            price - (a + b) / 2
            for price, a, b in zip(index_prices, spot1_prices, spot2_prices)
        ),
        "spot 2": statistics.mean(spot2_prices) - 300.0,
        "arb spot 2": sum(
            agent.get_asset_volume(market_id=spot2.market_id)
            for agent in arbitrage_agents
        )
        - initial,
        "arb index": sum(
            agent.get_asset_volume(market_id=index.market_id)
            for agent in arbitrage_agents
        )
        - initial,
    }


def compare(seeds: Sequence[int], main_steps: int = 100) -> Dict[str, Dict[str, float]]:
    """Run every case with every seed and average the measures.

    Args:
        seeds (Sequence[int]): seeds of the runs.
        main_steps (int): number of steps of the main session.

    Returns:
        Dict[str, Dict[str, float]]: for each case, the mean of each measure of
        :func:`measure` and the number of seeds where spot 2 is above 300 on
        average (``spot 2 up``).

    """
    results: Dict[str, Dict[str, float]] = {}
    for name, (spot_weight, index_weight) in CASES.items():
        rows: List[Dict[str, float]] = [
            measure(run_simulation(seed, spot_weight, index_weight, main_steps))
            for seed in seeds
        ]
        results[name] = {
            key: statistics.mean(row[key] for row in rows) for key in rows[0]
        }
        results[name]["spot 2 up"] = sum(1 for row in rows if row["spot 2"] > 0)
    return results


def print_comparison(results: Dict[str, Dict[str, float]], n_seeds: int) -> None:
    """Print one line per case.

    Args:
        results (Dict[str, Dict[str, float]]): the result of :func:`compare`.
        n_seeds (int): the number of seeds of each case.

    """
    print(
        f"{'case':<19} {'premium':>7} {'spot 2':>6} {'up':>5} "
        f"{'arb spot 2':>10} {'arb index':>9}"
    )
    for name, row in results.items():
        up = f"{row['spot 2 up']:.0f}/{n_seeds}"
        print(
            f"{name:<19} {row['premium']:7.2f} {row['spot 2']:6.2f} {up:>5} "
            f"{row['arb spot 2']:10.1f} {row['arb index']:9.1f}"
        )


def main() -> None:
    """Run the sample with seed 42, then compare the cases over ten seeds."""
    runner = run_simulation(SEED)
    print_prices(runner, times=[99, 100, 101, 150, 200, 300, 599])
    plot_prices(runner, path="shock_transfer_prices.png")
    print_comparison(compare(SEEDS), n_seeds=len(SEEDS))


if __name__ == "__main__":
    main()
