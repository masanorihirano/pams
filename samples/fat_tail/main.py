"""Fat tail sample.

Port of the FatTail sample of plhamJ. FCN agents without the chartist component (the
CI2002 model) trade in one market, and the returns of the market price show stylized
facts of real markets such as fat tails and volatility clustering.
``examples/fat_tail.ipynb`` analyses them. Each printed line has the same columns as
the output of plhamJ: session, time, market ID, market name, market price and
fundamental price.

``config.json`` is shortened so that the sample finishes in a few seconds. For the
statistics, run it at the scale of plhamJ: set ``iterationSteps`` of the second
session to ``60000`` and, as the analysis of plhamJ does, discard the first ``10000``
steps of that session before calculating the returns.

The config of plhamJ also sets ``"marginType": "normal"``. With normal margins, PAMS
chooses the side of an order differently from plhamJ (from the expected price and the
market price instead of from the sign of the random margin), and with this config the
market price then swings far away from the fundamental price. This sample therefore
uses the fixed margins of the CI2002 sample.
"""

import argparse
import random
from typing import Optional

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner


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
    runner.main()


if __name__ == "__main__":
    main()
