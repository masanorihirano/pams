"""Measure the time per step of the parallel runners on the parallel_main sample.

This is the counterpart of the measurement in the ParallelMain tutorial of Plham. For
each workload, runner and number of parallel workers, the first few steps of the
model of make_config.py are run and timed. For example::

    python -m samples.parallel_main.benchmark --output benchmark.csv --plot benchmark.png

Each step is split into two phases:

- collect: the normal agents decide their orders. This is T_w + T_c in the terms of
  Torii et al. (2017), i.e., the time of the agents' decisions and, for the parallel
  runners, the overhead of sending the agents and the simulation to the workers and
  receiving the orders.
- handle: the orders are executed and the high-frequency agents are asked. This is
  T_m of Torii et al. (2017), which is always processed sequentially.

The column "step" is the time of the two phases together, i.e., of ``_update_markets``
of the runner. It does not include the events and the updates of the markets between
the steps.

Each run is written to the CSV file as soon as it finishes. The runs must give the same
results. If not, RuntimeError is raised. ``--plot`` requires matplotlib. See ``--help``
for the options.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import time
import warnings
from io import TextIOWrapper
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from pams.logs.base import Logger
from pams.order import Cancel
from pams.order import Order
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners.sequential import SequentialRunner
from pams.session import Session
from pams.simulator import Simulator
from samples.parallel_main.make_config import make_config
from samples.parallel_main.workload_fcn_agent import WorkloadFCNAgent

Workload = Optional[Tuple[int, int]]

DEFAULT_NUM_PARALLEL: List[int] = [1, 2, 4, 8]
DEFAULT_RUNNERS: List[str] = ["sequential", "multi_process"]
DEFAULT_WORKLOADS: List[str] = ["off", "10x10", "100x100"]

COLUMNS: List[str] = [
    "workload",
    "runner",
    "numParallel",
    "repeat",
    "total",
    "first_step",
    "step",
    "collect",
    "handle",
    "digest",
]


class StepTimer(SequentialRunner):
    """Runner that records the time of each step and of its phases.

    The parallel runners are timed by the subclasses that also inherit from them.
    """

    def __init__(
        self,
        settings: Union[Dict, TextIOWrapper, os.PathLike, str],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator,
    ):
        """Initialize.

        Args:
            settings (Union[Dict, TextIOWrapper, os.PathLike, str]): runner configuration.
            prng (random.Random, Optional): pseudo random number generator for this runner.
            logger (Logger, Optional): logger instance.
            simulator_class (Type[Simulator]): type of simulator.

        Returns:
            None

        """
        super().__init__(settings, prng, logger, simulator_class)
        self.step_times: List[float] = []
        self.collect_times: List[float] = []
        self.handle_times: List[float] = []

    def _update_markets(self, session: Session) -> None:
        start: float = time.perf_counter()
        super()._update_markets(session=session)
        self.step_times.append(time.perf_counter() - start)

    def _collect_orders_from_normal_agents(
        self, session: Session
    ) -> List[List[Union[Order, Cancel]]]:
        start: float = time.perf_counter()
        orders = super()._collect_orders_from_normal_agents(session=session)
        self.collect_times.append(time.perf_counter() - start)
        return orders

    def _handle_orders(
        self, session: Session, local_orders: List[List[Union[Order, Cancel]]]
    ) -> List[List[Union[Order, Cancel]]]:
        start: float = time.perf_counter()
        orders = super()._handle_orders(session=session, local_orders=local_orders)
        self.handle_times.append(time.perf_counter() - start)
        return orders


class TimedMultiThreadRunner(StepTimer, MultiThreadAgentParallelRunner):
    """MultiThreadAgentParallelRunner that records the time of each step."""


class TimedMultiProcessRunner(StepTimer, MultiProcessAgentParallelRunner):
    """MultiProcessAgentParallelRunner that records the time of each step."""


RUNNERS: Dict[str, Type[StepTimer]] = {
    "sequential": StepTimer,
    "multi_thread": TimedMultiThreadRunner,
    "multi_process": TimedMultiProcessRunner,
}


def parse_workload(value: str) -> Workload:
    """Parse a workload given as "off" or "<bsNumSamples>x<bsNumSteps>".

    Args:
        value (str): workload.

    Returns:
        Workload: None for "off", otherwise the numbers of paths and time steps.

    """
    if value == "off":
        return None
    parts: List[str] = value.split("x")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        n_samples, n_steps = int(parts[0]), int(parts[1])
        if n_samples > 0 and n_steps > 0:
            return n_samples, n_steps
    raise argparse.ArgumentTypeError(
        f'workload has to be "off" or "<bsNumSamples>x<bsNumSteps>", not "{value}"'
    )


def format_workload(workload: Workload) -> str:
    """Format a workload in the same way as :func:`parse_workload` reads it.

    Args:
        workload (Workload): workload.

    Returns:
        str: "off" or "<bsNumSamples>x<bsNumSteps>".

    """
    if workload is None:
        return "off"
    return f"{workload[0]}x{workload[1]}"


def result_digest(runner: SequentialRunner) -> str:
    """Summarize the results of a simulation into a digest.

    Args:
        runner (SequentialRunner): runner after the simulation.

    Returns:
        str: SHA-256 digest of the market prices, the fundamental prices and the
        executed volumes of all the markets at all the times.

    """
    series: List[List[Any]] = []
    for market in runner.simulator.markets:
        times = range(market.get_time() + 1)
        series.append(market.get_market_prices(times=times))
        series.append(market.get_fundamental_prices(times=times))
        series.append(market.get_executed_volumes(times=times))
    return hashlib.sha256(json.dumps(series).encode("utf-8")).hexdigest()


def run_one(
    settings: Dict[str, Any], runner_name: str, seed: int, warmup: int
) -> Dict[str, Any]:
    """Run a simulation and measure the time.

    Args:
        settings (Dict[str, Any]): config.
        runner_name (str): name of the runner in :data:`RUNNERS`.
        seed (int): seed of the simulation.
        warmup (int): number of steps excluded from the mean times per step.

    Returns:
        Dict[str, Any]: the total time, the time of the first step, the mean times of
        a step and its phases after the warmup, and the digest of the results.

    """
    runner = RUNNERS[runner_name](
        settings=settings,
        prng=random.Random(seed),  # nosec B311 # a simulation, not for security
        logger=None,
    )
    runner.class_register(cls=WorkloadFCNAgent)
    # runner.main() without printing the time
    start: float = time.perf_counter()
    runner._setup()
    runner._run()
    total: float = time.perf_counter() - start
    return {
        "total": total,
        "first_step": runner.step_times[0],
        "step": statistics.mean(runner.step_times[warmup:]),
        "collect": statistics.mean(runner.collect_times[warmup:]),
        "handle": statistics.mean(runner.handle_times[warmup:]),
        "digest": result_digest(runner=runner),
    }


def check_digests(rows: List[Dict[str, Any]]) -> None:
    """Check that all the runs give the same results.

    Args:
        rows (List[Dict[str, Any]]): results of the runs.

    Returns:
        None

    """
    if len({row["digest"] for row in rows}) > 1:
        lines: List[str] = [
            f"{row['workload']} {row['runner']} {row['numParallel']} "
            f"{row['repeat']}: {row['digest']}"
            for row in rows
        ]
        raise RuntimeError("the results differ between the runs:\n" + "\n".join(lines))


def write_rows(path: str, rows: List[Dict[str, Any]], append: bool) -> None:
    """Write the results of runs to a CSV file.

    Args:
        path (str): output CSV file.
        rows (List[Dict[str, Any]]): results of the runs.
        append (bool): whether the rows are appended to the file. If False, the file
            is overwritten with the header and the rows.

    Returns:
        None

    """
    with open(path, "a" if append else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _format_major_tick(value: float, _: Any) -> str:
    """Format a major tick of a logarithmic axis without exponents (internal usage).

    Args:
        value (float): value of the tick.
        _ (Any): position of the tick (not used).

    Returns:
        str: label of the tick.

    """
    return f"{value:g}"


def _format_minor_tick(value: float, _: Any) -> str:
    """Format a minor tick of a logarithmic axis at 2 and 5 times powers of 10 only.

    Args:
        value (float): value of the tick.
        _ (Any): position of the tick (not used).

    Returns:
        str: label of the tick, empty for the other ticks.

    """
    if value <= 0.0:
        return ""
    mantissa: float = value / 10 ** math.floor(math.log10(value))
    if round(mantissa) in (2, 5):
        return f"{value:g}"
    return ""


def plot(rows: List[Dict[str, Any]], path: str, title: str) -> None:
    """Plot the time per step against the number of parallel workers.

    One panel is drawn for each workload. The median over the repeats is plotted.
    The times of the sequential runner are drawn as dashed horizontal lines.

    Args:
        rows (List[Dict[str, Any]]): results of the runs.
        path (str): output image file.
        title (str): title of the figure.

    Returns:
        None

    """
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.ticker import FuncFormatter  # pylint: disable=C0415

    workloads: List[str] = list(dict.fromkeys(row["workload"] for row in rows))
    runner_names: List[str] = list(dict.fromkeys(row["runner"] for row in rows))
    parallel_runners: List[str] = [n for n in runner_names if n != "sequential"]
    num_parallels: List[int] = sorted(
        {row["numParallel"] for row in rows if row["runner"] != "sequential"}
    )
    phases: List[Tuple[str, str, str]] = [
        ("step", "step", "tab:red"),
        ("collect", "collect (T_w + T_c)", "tab:green"),
        ("handle", "handle (T_m)", "tab:blue"),
    ]
    line_styles: List[str] = ["-", "-.", ":"]
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(4.2 * len(workloads), 4.2), squeeze=False
    )
    for ax, workload in zip(axes[0], workloads):
        for key, label, color in phases:
            for runner_name, line_style in zip(parallel_runners, line_styles):
                points: List[Tuple[int, float]] = [
                    (
                        num_parallel,
                        statistics.median(
                            row[key]
                            for row in rows
                            if row["workload"] == workload
                            and row["runner"] == runner_name
                            and row["numParallel"] == num_parallel
                        ),
                    )
                    for num_parallel in num_parallels
                ]
                ax.plot(
                    [x for x, _ in points],
                    [y for _, y in points],
                    marker="o",
                    linestyle=line_style,
                    color=color,
                    label=f"{label}, {runner_name}",
                )
            sequential: List[float] = [
                row[key]
                for row in rows
                if row["workload"] == workload and row["runner"] == "sequential"
            ]
            if len(sequential) > 0:
                ax.axhline(
                    statistics.median(sequential),
                    linestyle="--",
                    color=color,
                    label=f"{label}, sequential",
                )
        if len(num_parallels) > 0:
            ax.set_xscale("log", base=2)
            ax.set_xticks(num_parallels)
            ax.set_xticklabels([str(n) for n in num_parallels])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_format_major_tick))
        ax.yaxis.set_minor_formatter(FuncFormatter(_format_minor_tick))
        ax.set_xlabel("numParallel")
        ax.set_ylabel("time per step [s]")
        ax.set_title(f"workload {workload}")
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(phases), fontsize="small")
    fig.suptitle(title)
    n_legend_rows: int = len(parallel_runners) + 1
    fig.tight_layout(rect=(0.0, 0.035 * (n_legend_rows + 1), 1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)


def run_all(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run the simulations with all the combinations of the options.

    Args:
        args (argparse.Namespace): options of :func:`main`.

    Returns:
        List[Dict[str, Any]]: results of the runs.

    """
    rows: List[Dict[str, Any]] = []
    for repeat in range(args.repeat):
        for workload in args.workloads:
            for runner_name in args.runners:
                num_parallels: List[int] = (
                    [0] if runner_name == "sequential" else args.num_parallel
                )
                for num_parallel in num_parallels:
                    settings: Dict[str, Any] = make_config(
                        n_spots=args.spots,
                        steps=args.steps,
                        agents_per_market=args.agents_per_market,
                        arbitrage_agents=args.arbitrage_agents,
                        order_rate=args.order_rate,
                        with_workload=workload is not None,
                        bs_num_samples=workload[0] if workload is not None else 10,
                        bs_num_steps=workload[1] if workload is not None else 10,
                        num_parallel=num_parallel if num_parallel > 0 else None,
                    )
                    row: Dict[str, Any] = {
                        "workload": format_workload(workload),
                        "runner": runner_name,
                        "numParallel": num_parallel,
                        "repeat": repeat,
                    }
                    row.update(
                        run_one(
                            settings=settings,
                            runner_name=runner_name,
                            seed=args.seed,
                            warmup=args.warmup,
                        )
                    )
                    rows.append(row)
                    if args.output is not None:
                        write_rows(path=args.output, rows=[row], append=True)
                    print(
                        f"workload={row['workload']} runner={runner_name} "
                        f"numParallel={num_parallel} repeat={repeat} "
                        f"total={row['total']:.3f}s "
                        f"first_step={row['first_step']:.3f}s "
                        f"step={row['step']:.4f}s "
                        f"collect={row['collect']:.4f}s "
                        f"handle={row['handle']:.4f}s "
                        f"digest={row['digest'][:12]}",
                        flush=True,
                    )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure the time per step of the parallel runners."
    )
    parser.add_argument(
        "--spots", type=int, default=9, help="number of spot markets (default: 9)"
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="number of steps (default: 5)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of first steps excluded from the mean times per step "
        "(default: 1)",
    )
    parser.add_argument(
        "--agents-per-market",
        type=int,
        default=500,
        help="number of FCN agents for each market (default: 500)",
    )
    parser.add_argument(
        "--arbitrage-agents",
        type=int,
        default=100,
        help="number of arbitrage agents (default: 100)",
    )
    parser.add_argument(
        "--order-rate",
        type=float,
        default=0.1,
        help="probability that an FCN agent submits orders (default: 0.1)",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_PARALLEL,
        help="numbers of parallel workers "
        f"(default: {' '.join(map(str, DEFAULT_NUM_PARALLEL))})",
    )
    parser.add_argument(
        "--runners",
        nargs="+",
        choices=list(RUNNERS.keys()),
        default=DEFAULT_RUNNERS,
        help=f"runners (default: {' '.join(DEFAULT_RUNNERS)})",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=DEFAULT_WORKLOADS,
        help='workloads, "off" or "<bsNumSamples>x<bsNumSteps>" '
        f"(default: {' '.join(DEFAULT_WORKLOADS)})",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="number of repeats (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="simulation random seed (default: 1)"
    )
    parser.add_argument("--output", type=str, default=None, help="output CSV file")
    parser.add_argument("--plot", type=str, default=None, help="output image file")
    args = parser.parse_args(argv)
    if args.steps <= args.warmup:
        parser.error("--steps has to be larger than --warmup")
    try:
        args.workloads = [parse_workload(value) for value in args.workloads]
    except argparse.ArgumentTypeError as e:
        parser.error(f"argument --workloads: {e}")
    if args.plot is not None and importlib.util.find_spec("matplotlib") is None:
        parser.error("--plot requires matplotlib")

    if args.output is not None:
        write_rows(path=args.output, rows=[], append=False)
    with warnings.catch_warnings():
        # the parallel runners warn that they are experimental whenever they are made
        warnings.filterwarnings("ignore", message=".* is experimental")
        rows: List[Dict[str, Any]] = run_all(args=args)
    check_digests(rows=rows)
    if args.plot is not None:
        n_agents: int = (args.spots + 1) * args.agents_per_market
        plot(
            rows=rows,
            path=args.plot,
            title=(
                f"{args.spots} spot markets + 1 index market, {n_agents} FCN agents, "
                f"{args.arbitrage_agents} arbitrage agents, "
                f"mean of steps {args.warmup + 1}-{args.steps}"
            ),
        )


if __name__ == "__main__":
    main()
