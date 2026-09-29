import argparse
import random
from typing import Dict
from typing import Optional
from typing import Type

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners.sequential import SequentialRunner
from samples.black_scholes.workload_fcn_agent import WorkloadFCNAgent

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
        default="sequential",
        help="simulation runner",
    )
    args = parser.parse_args()
    config: str = args.config
    seed: Optional[int] = args.seed
    runner_class: Type[SequentialRunner] = RUNNERS[args.runner]

    runner = runner_class(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,
        logger=MarketStepPrintLogger(),
    )
    runner.class_register(cls=WorkloadFCNAgent)
    runner.main()


if __name__ == "__main__":
    main()
