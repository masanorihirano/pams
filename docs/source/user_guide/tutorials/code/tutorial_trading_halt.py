"""Tutorial: trading halts after a fall of the fundamental price."""
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

from pams.events import TradingHaltRule
from pams.logs import Logger
from pams.logs import MarketStepEndLog
from pams.runners import SequentialRunner

SEED: int = 42
SEEDS: Tuple[int, ...] = tuple(range(42, 52))

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
                "events": ["FundamentalPriceShock", "TradingHaltRule"],
            },
        ],
    },
    "FundamentalPriceShock": {
        "class": "FundamentalPriceShock",
        "target": "Market",
        "triggerTime": 0,
        "priceChangeRate": -0.1,
        "enabled": True,
    },
    "TradingHaltRule": {
        "class": "TradingHaltRule",
        "targetMarkets": ["Market"],
        "triggerChangeRate": 0.05,
        "haltingTimeLength": 100,
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
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "noiseScale": 0.001,
        "timeWindowSize": [100, 200],
        "orderMargin": [0.0, 0.1],
    },
}
# [config-end]

# [cases-start]
# the threshold, the length of a halt and whether the rule is on
CASES: Dict[str, Tuple[float, int, bool]] = {
    "halt 5%, 100 steps": (0.05, 100, True),
    "no halt": (0.05, 100, False),
    "halt 5%, 200 steps": (0.05, 200, True),
    "halt 2%, 100 steps": (0.02, 100, True),
}
# [cases-end]


# [recorder-start]
class HaltRecorder(Logger):
    """Record the steps at whose end the market is halted."""

    def __init__(self) -> None:
        """Initialize the recorder."""
        super().__init__()
        self.halted: List[int] = []

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """Keep the step if the market is not running at its end.

        Args:
            log (MarketStepEndLog): the log of the end of a market step.

        """
        # in a session without order execution, the market never runs
        if log.session.with_order_execution and not log.market.is_running:
            self.halted.append(log.market.get_time())


# [recorder-end]


def run_simulation(
    seed: int,
    rate: float = 0.05,
    length: int = 100,
    enabled: bool = True,
    main_steps: int = 500,
) -> Tuple[SequentialRunner, HaltRecorder]:
    """Run the simulation with the given trading halt rule.

    Args:
        seed (int): seed of the random number generator.
        rate (float): threshold ``triggerChangeRate`` of the rule.
        length (int): length ``haltingTimeLength`` of a halt.
        enabled (bool): whether the rule is on.
        main_steps (int): number of steps of the main session.

    Returns:
        Tuple[SequentialRunner, HaltRecorder]: the runner after the run and the
        recorder of the halted steps.

    """
    config = copy.deepcopy(CONFIG)
    config["TradingHaltRule"]["triggerChangeRate"] = rate
    config["TradingHaltRule"]["haltingTimeLength"] = length
    config["TradingHaltRule"]["enabled"] = enabled
    config["simulation"]["sessions"][1]["iterationSteps"] = main_steps
    recorder = HaltRecorder()
    runner = SequentialRunner(
        settings=config, prng=random.Random(seed), logger=recorder
    )
    # main() prints the time taken, which is not needed here
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    return runner, recorder


def halt_starts(halted: List[int], length: int) -> List[int]:
    """Find the steps at which the halts start.

    A halt that starts at step ``s`` covers the ends of the steps ``s`` to
    ``s + length``, so a halted step after them belongs to a new halt.

    Args:
        halted (List[int]): the halted steps, in order.
        length (int): length ``haltingTimeLength`` of a halt.

    Returns:
        List[int]: the first step of each halt.

    """
    starts: List[int] = []
    for time in halted:
        if len(starts) == 0 or time > starts[-1] + length:
            starts.append(time)
    return starts


def halt_stats(runner: SequentialRunner, recorder: HaltRecorder) -> Dict[str, Any]:
    """Summarize the halts and the prices of the main session.

    Args:
        runner (SequentialRunner): the runner after the run.
        recorder (HaltRecorder): the recorder used as the logger of the run.

    Returns:
        Dict[str, Any]: the number of halts, the number of halted steps, the first
        step of each halt, the lowest and the last market prices, and the number
        of shares traded.

    """
    market = runner.simulator.name2market["Market"]
    main_session = runner.simulator.sessions[1]
    start = main_session.session_start_time
    times = range(start, start + main_session.iteration_steps)
    prices: List[float] = market.get_market_prices(times=times)
    # a disabled event registers no hooks and is not in name2event
    event = runner.simulator.name2event.get("TradingHaltRule")
    halts = 0
    starts: List[int] = []
    if isinstance(event, TradingHaltRule):
        halts = event.activation_count
        starts = halt_starts(recorder.halted, event.halting_time_length)
    return {
        "halts": halts,
        "halted": len(recorder.halted),
        "starts": starts,
        "lowest": min(prices),
        "last": prices[-1],
        "traded": sum(market.get_executed_volumes(times=times)),
    }


def run_cases(seed: int) -> Dict[str, Tuple[SequentialRunner, HaltRecorder]]:
    """Run every case with the same seed.

    Args:
        seed (int): seed of the random number generator.

    Returns:
        Dict[str, Tuple[SequentialRunner, HaltRecorder]]: the runner and the
        recorder of each case, by name.

    """
    return {
        name: run_simulation(seed, rate=rate, length=length, enabled=enabled)
        for name, (rate, length, enabled) in CASES.items()
    }


def print_cases(runs: Dict[str, Tuple[SequentialRunner, HaltRecorder]]) -> None:
    """Print the halts and the prices of each case.

    Args:
        runs (Dict[str, Tuple[SequentialRunner, HaltRecorder]]): the output of
            ``run_cases``.

    """
    print("case                halts  halted  lowest    last  traded  halt starts")
    for name, (runner, recorder) in runs.items():
        stats = halt_stats(runner, recorder)
        starts = ", ".join(str(time) for time in stats["starts"]) or "-"
        print(
            f"{name:<18}  {stats['halts']:5d}  {stats['halted']:6d}"
            f"  {stats['lowest']:6.2f}  {stats['last']:6.2f}"
            f"  {stats['traded']:6d}  {starts}"
        )


def compare(
    seeds: Sequence[int] = SEEDS, main_steps: int = 500
) -> Dict[str, List[Dict[str, Any]]]:
    """Run every case with each seed.

    Args:
        seeds (Sequence[int]): seeds of the runs.
        main_steps (int): number of steps of the main session.

    Returns:
        Dict[str, List[Dict[str, Any]]]: the statistics of each run, by case.

    """
    return {
        name: [
            halt_stats(
                *run_simulation(
                    seed,
                    rate=rate,
                    length=length,
                    enabled=enabled,
                    main_steps=main_steps,
                )
            )
            for seed in seeds
        ]
        for name, (rate, length, enabled) in CASES.items()
    }


def print_comparison(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print the means of the statistics over the seeds.

    Args:
        results (Dict[str, List[Dict[str, Any]]]): the output of ``compare``.

    """
    print("case                halts  halted  lowest    last  traded")
    for name, runs in results.items():
        means = {
            key: statistics.mean(run[key] for run in runs)
            for key in ["halts", "halted", "lowest", "last", "traded"]
        }
        print(
            f"{name:<18}  {means['halts']:5.1f}  {means['halted']:6.1f}"
            f"  {means['lowest']:6.2f}  {means['last']:6.2f}  {means['traded']:6.1f}"
        )


def plot_cases(
    runs: Dict[str, Tuple[SequentialRunner, HaltRecorder]], path: str
) -> None:
    """Plot the prices of each case, with the halts shaded.

    Args:
        runs (Dict[str, Tuple[SequentialRunner, HaltRecorder]]): the output of
            ``run_cases``.
        path (str): path of the image file to write.

    """
    times = range(100, 600)
    fig, axes = plt.subplots(len(runs), 1, sharex=True, sharey=True, figsize=(6.4, 8.0))
    for ax, (name, (runner, recorder)) in zip(axes, runs.items()):
        market = runner.simulator.name2market["Market"]
        ax.plot(
            times,
            market.get_market_prices(times=times),
            color="tab:red",
            label="market price",
        )
        ax.plot(
            times,
            market.get_fundamental_prices(times=times),
            color="black",
            label="fundamental price",
        )
        for start in halt_starts(recorder.halted, CASES[name][1]):
            end = min(start + CASES[name][1] + 1, times[-1])
            ax.axvspan(start, end, color="tab:blue", alpha=0.2, linewidth=0)
        ax.set_title(name, fontsize="medium")
        ax.set_ylabel("price")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    """Run the experiments of the tutorial and plot the prices of seed 42."""
    runs = run_cases(SEED)
    print_cases(runs)
    print_comparison(compare())
    plot_cases(runs, path="trading_halt_prices.png")


if __name__ == "__main__":
    main()
