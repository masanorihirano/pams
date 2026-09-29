"""Run the invest_div sample simulation.

This sample is a port of the ``InvestDiv`` sample of plhamJ. After a fundamental price shock in Market-1, the
:class:`samples.invest_div.invest_div_fcn_agent.InvestDivFCNAgent` agents reduce their positions under the
regulation for investment diversification.

``InvestDivFCNAgent`` is imported from the ``samples.invest_div`` package, so run this sample as a module from the
root of the repository (or add the root to ``PYTHONPATH``)::

    python -m samples.invest_div.main --config samples/invest_div/config.json --seed 1

"""

import argparse
import random
from typing import Optional

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner
from samples.invest_div.invest_div_fcn_agent import InvestDivFCNAgent


def main() -> None:
    """Run the simulation with the config file and the seed given as command line arguments."""
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
    runner.class_register(cls=InvestDivFCNAgent)
    runner.main()


if __name__ == "__main__":
    main()
