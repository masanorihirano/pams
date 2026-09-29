"""Cancel test sample.

Port of the CancelTest sample of plhamJ: FCN agents (``CancelFCNAgent``) cancel each of
their orders with the probability ``cancelRate`` right after submitting it.

``CancelFCNAgent`` is imported from the ``samples.cancel_test`` package, so run this
sample as a module from the root of the repository (or add the root to ``PYTHONPATH``)::

    python -m samples.cancel_test.main --config samples/cancel_test/config.json --seed 1
"""
import argparse
import random
from typing import Optional

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner
from samples.cancel_test.cancel_fcn_agent import CancelFCNAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="config.json file"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="simulation random seed"
    )
    args = parser.parse_args()
    config: str = args.config
    seed: Optional[int] = args.seed

    runner = SequentialRunner(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,
        logger=MarketStepPrintLogger(),
    )
    runner.class_register(cls=CancelFCNAgent)
    runner.main()


if __name__ == "__main__":
    main()
