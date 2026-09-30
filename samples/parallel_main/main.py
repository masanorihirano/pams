"""Run the shock transfer model with many spot markets by a parallel runner.

This is the counterpart of ParallelMain of Plham. Only the runner is chosen here; the
model is given by the config. For example::

    python -m samples.parallel_main.main --config samples/parallel_main/config.json --seed 1
    python -m samples.parallel_main.main -c samples/parallel_main/config-009.json -s 1 -n 4

For the same seed, all the runners print the same market log. Only the lines starting
with ``#``, which report the time taken, differ.
"""

import argparse
import json
import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners.sequential import SequentialRunner
from samples.parallel_main.workload_fcn_agent import WorkloadFCNAgent

RUNNERS: Dict[str, Type[SequentialRunner]] = {
    "sequential": SequentialRunner,
    "multi_thread": MultiThreadAgentParallelRunner,
    "multi_process": MultiProcessAgentParallelRunner,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="config.json file"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="simulation random seed"
    )
    parser.add_argument(
        "--runner",
        "-r",
        type=str,
        choices=list(RUNNERS.keys()),
        default="multi_process",
        help="simulation runner",
    )
    parser.add_argument(
        "--num-parallel",
        "-n",
        type=int,
        default=None,
        help="number of parallel workers (overrides numParallel of the config)",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        settings: Dict[str, Any] = json.load(f)
    if args.num_parallel is not None:
        settings["simulation"]["numParallel"] = args.num_parallel
    seed: Optional[int] = args.seed
    runner_class: Type[SequentialRunner] = RUNNERS[args.runner]

    runner = runner_class(
        settings=settings,
        prng=random.Random(seed) if seed is not None else None,  # nosec B311
        logger=MarketStepPrintLogger(),
    )
    runner.class_register(cls=WorkloadFCNAgent)
    runner.main()


if __name__ == "__main__":
    main()
