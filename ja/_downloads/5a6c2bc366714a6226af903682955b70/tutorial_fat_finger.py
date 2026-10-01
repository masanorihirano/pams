"""Tutorial: a fat-finger order, a mistaken large sell order, and the order book."""
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
from typing import Tuple
from typing import cast

import matplotlib.pyplot as plt

from pams.events import OrderMistakeShock
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
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
                "events": ["OrderMistakeShock"],
            },
        ],
    },
    "OrderMistakeShock": {
        "class": "OrderMistakeShock",
        "target": "Market",
        "triggerTime": 100,
        "priceChangeRate": -0.05,
        "orderVolume": 10000,
        "orderTimeLength": 10000,
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


# [recorder-start]
class BookRecorder(Logger):
    """Keep every order and execution, and the order book at the end of every step."""

    def __init__(self) -> None:
        """Initialize the recorder."""
        super().__init__()
        self.orders: List[OrderLog] = []
        self.executions: List[ExecutionLog] = []
        self.buy_books: Dict[int, Dict[Optional[float], int]] = {}
        self.sell_books: Dict[int, Dict[Optional[float], int]] = {}

    def process_order_log(self, log: OrderLog) -> None:
        """Keep an order accepted by the market.

        Args:
            log (OrderLog): the order, after the market has rounded its price.

        """
        self.orders.append(log)

    def process_execution_log(self, log: ExecutionLog) -> None:
        """Keep an execution.

        Args:
            log (ExecutionLog): the execution.

        """
        self.executions.append(log)

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """Keep the volume at each price of the order book at the end of the step.

        Args:
            log (MarketStepEndLog): the log of the market at the end of the step.

        """
        time: int = log.market.get_time()
        self.buy_books[time] = log.market.get_buy_order_book()
        self.sell_books[time] = log.market.get_sell_order_book()


# [recorder-end]


def run_simulation(
    seed: int, enabled: bool = True, volume: Optional[int] = None
) -> Tuple[SequentialRunner, BookRecorder]:
    """Run the simulation, with or without the mistaken order.

    Args:
        seed (int): seed of the random number generator.
        enabled (bool): whether the mistaken order is placed.
        volume (int, Optional): volume of the mistaken order. If None, the volume of
            the configuration (10000) is used.

    Returns:
        Tuple[SequentialRunner, BookRecorder]: the runner after the run and the
        recorder used as its logger.

    """
    config: Dict[str, Any] = copy.deepcopy(CONFIG)
    config["OrderMistakeShock"]["enabled"] = enabled
    if volume is not None:
        config["OrderMistakeShock"]["orderVolume"] = volume
    recorder = BookRecorder()
    runner = SequentialRunner(
        settings=config, prng=random.Random(seed), logger=recorder
    )
    # hide the two time lines printed by main()
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    return runner, recorder


# [wall-start]
def find_mistake(runner: SequentialRunner, recorder: BookRecorder) -> OrderLog:
    """Find the mistaken order among the orders of the run.

    Args:
        runner (SequentialRunner): the runner after the run.
        recorder (BookRecorder): the recorder of the run.

    Returns:
        OrderLog: the order placed at the trigger step. Only one order is placed per
        step, so this is the order that the event replaced.

    """
    event = cast(OrderMistakeShock, runner.simulator.name2event["OrderMistakeShock"])
    return next(order for order in recorder.orders if order.time == event.trigger_time)


def wall_left(recorder: BookRecorder, mistake: OrderLog, time: int) -> int:
    """Count the shares of the mistaken order not executed at the end of a step.

    Args:
        recorder (BookRecorder): the recorder of the run.
        mistake (OrderLog): the mistaken order.
        time (int): the step.

    Returns:
        int: the volume of the mistaken order minus its executions up to ``time``.

    """
    executed: int = sum(
        execution.volume
        for execution in recorder.executions
        if execution.sell_order_id == mistake.order_id and execution.time <= time
    )
    return mistake.volume - executed


# [wall-end]


def print_shock(
    runner: SequentialRunner,
    recorder: BookRecorder,
    times: Sequence[int] = (199, 200, 201, 250, 400, 599),
) -> None:
    """Print the mistaken order, and the market and its order book around it.

    Args:
        runner (SequentialRunner): the runner after the run.
        recorder (BookRecorder): the recorder of the run.
        times (Sequence[int]): the steps to print.

    """
    market = runner.simulator.name2market["Market"]
    mistake: OrderLog = find_mistake(runner, recorder)
    owner: str = runner.simulator.id2agent[mistake.agent_id].name
    side: str = "buys" if mistake.is_buy else "sells"
    print(
        f"mistaken order: {owner} {side} {mistake.volume} at {mistake.price:.5f} "
        f"in step {mistake.time}"
    )
    print("step    price  best buy  best sell  buy volume  sell volume  wall left")
    for time in times:
        buy_book = recorder.buy_books[time]
        sell_book = recorder.sell_books[time]
        left: str = (
            str(wall_left(recorder, mistake, time)) if time >= mistake.time else "-"
        )
        print(
            f"{time:4d}  {market.get_market_price(time):7.3f}"
            f"  {max(price for price in buy_book if price is not None):8.3f}"
            f"  {min(price for price in sell_book if price is not None):9.3f}"
            f"  {sum(buy_book.values()):10d}  {sum(sell_book.values()):11d}"
            f"  {left:>9}"
        )


def price_stats(runner: SequentialRunner, start: int, end: int) -> Dict[str, float]:
    """Summarize the market prices and the volume of some steps.

    Args:
        runner (SequentialRunner): the runner after the run.
        start (int): the first step.
        end (int): the step after the last one.

    Returns:
        Dict[str, float]: the lowest, highest and last market prices, and the traded
        volume.

    """
    market = runner.simulator.name2market["Market"]
    prices: List[float] = market.get_market_prices(times=range(start, end))
    return {
        "min": min(prices),
        "max": max(prices),
        "last": prices[-1],
        "volume": sum(market.get_executed_volumes(times=range(start, end))),
    }


def print_comparison(
    runners: Dict[str, SequentialRunner], start: int = 201, end: int = 600
) -> None:
    """Print the prices and the volume of some steps for each run.

    Args:
        runners (Dict[str, SequentialRunner]): the runners after the run, by name.
        start (int): the first step.
        end (int): the step after the last one.

    """
    print(f"steps {start}-{end - 1}        min      max     last  volume")
    for name, runner in runners.items():
        stats: Dict[str, float] = price_stats(runner, start=start, end=end)
        print(
            f"{name:<17}  {stats['min']:7.3f}  {stats['max']:7.3f}"
            f"  {stats['last']:7.3f}  {stats['volume']:6d}"
        )


# [volumes-start]
def compare_volumes(
    volumes: Sequence[Optional[int]] = (10000, 100, 10, None),
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, List[Tuple[Optional[int], float]]]:
    """Run the simulation with mistaken orders of several volumes, and without one.

    Args:
        volumes (Sequence[int, Optional]): the volumes of the mistaken order. None
            runs the simulation without the mistaken order.
        seeds (Sequence[int]): the seeds of the runs.

    Returns:
        Dict[str, List[Tuple[int, Optional], float]]: for each volume, and for each
        seed, the step at which the mistaken order is fully executed (None if it is
        not) and the mean market price of steps 500 to 599.

    """
    results: Dict[str, List[Tuple[Optional[int], float]]] = {}
    for volume in volumes:
        rows: List[Tuple[Optional[int], float]] = []
        for seed in seeds:
            runner, recorder = run_simulation(
                seed, enabled=volume is not None, volume=volume
            )
            used_up: Optional[int] = None
            if volume is not None:
                mistake: OrderLog = find_mistake(runner, recorder)
                used_up = next(
                    (
                        time
                        for time in range(mistake.time, 600)
                        if wall_left(recorder, mistake, time) == 0
                    ),
                    None,
                )
            market = runner.simulator.name2market["Market"]
            mean_price: float = statistics.mean(
                market.get_market_prices(times=range(500, 600))
            )
            rows.append((used_up, mean_price))
        results["none" if volume is None else str(volume)] = rows
    return results


# [volumes-end]


def print_volumes(results: Dict[str, List[Tuple[Optional[int], float]]]) -> None:
    """Print how often and when the mistaken order is used up, and the late prices.

    Args:
        results (Dict[str, List[Tuple[int, Optional], float]]): the results of
            :func:`compare_volumes`.

    """
    print("volume  used up  mean step  mean price 500-599")
    for name, rows in results.items():
        steps: List[int] = [used_up for used_up, _ in rows if used_up is not None]
        used: str = "-" if name == "none" else f"{len(steps)}/{len(rows)}"
        step: str = f"{statistics.mean(steps):.1f}" if steps else "-"
        price: float = statistics.mean(mean_price for _, mean_price in rows)
        print(f"{name:>6}  {used:>7}  {step:>9}  {price:18.2f}")


def plot_prices(runners: Dict[str, SequentialRunner], wall: float, path: str) -> None:
    """Plot the market prices of the main session of each run.

    Args:
        runners (Dict[str, SequentialRunner]): the runners after the run, by name.
            The first one gives the fundamental price.
        wall (float): the price of the mistaken order.
        path (str): path of the image file to write.

    """
    times: List[int] = list(range(100, 600))
    # draw the second run over the third one
    styles: List[Tuple[str, int]] = [("#9e9e9e", 1), ("#2a78d6", 3), ("#eb6834", 2)]
    fig, ax = plt.subplots()
    for (name, runner), (color, zorder) in zip(runners.items(), styles):
        market = runner.simulator.name2market["Market"]
        ax.plot(
            times,
            market.get_market_prices(times=times),
            color=color,
            zorder=zorder,
            label=name,
        )
    market = next(iter(runners.values())).simulator.name2market["Market"]
    ax.plot(
        times,
        market.get_fundamental_prices(times=times),
        color="black",
        label="fundamental price",
    )
    ax.axhline(wall, color="black", linestyle="--", linewidth=1, label="mistaken order")
    ax.set_xlabel("step")
    ax.set_ylabel("price")
    # keep the legend below the axes, away from the lines
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_book(
    recorder: BookRecorder, path: str, start: int = 150, end: int = 351
) -> None:
    """Plot the prices of the buy and sell orders in the order book at each step.

    Args:
        recorder (BookRecorder): the recorder of the run.
        path (str): path of the image file to write.
        start (int): the first step.
        end (int): the step after the last one.

    """
    fig, ax = plt.subplots()
    for books, color, label in [
        (recorder.sell_books, "#eb6834", "sell orders"),
        (recorder.buy_books, "#2a78d6", "buy orders"),
    ]:
        points: List[Tuple[int, float]] = [
            (time, price)
            for time in range(start, end)
            for price in books[time]
            if price is not None
        ]
        ax.scatter(
            [time for time, _ in points],
            [price for _, price in points],
            s=2,
            color=color,
            label=label,
        )
    ax.set_ylim(265, 305)
    ax.set_xlabel("step")
    ax.set_ylabel("price")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, markerscale=4)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run all the steps of the tutorial."""
    runner, recorder = run_simulation(SEED)
    print_shock(runner, recorder)
    runners: Dict[str, SequentialRunner] = {
        "no mistake": run_simulation(SEED, enabled=False)[0],
        "mistake, 10000": runner,
        "mistake, 100": run_simulation(SEED, volume=100)[0],
    }
    print_comparison(runners)
    print_volumes(compare_volumes())
    wall: float = cast(float, find_mistake(runner, recorder).price)
    plot_prices(runners, wall=wall, path="fat_finger_prices.png")
    plot_book(recorder, path="fat_finger_book.png")


if __name__ == "__main__":
    main()
