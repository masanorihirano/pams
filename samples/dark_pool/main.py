"""Dark pool sample.

Port of the DarkPool sample of plhamJ, based on Mizuta et al. (2015). FCN agents trade
in a lit market and in a dark pool
(:class:`samples.dark_pool.dark_pool_market.DarkPoolMarket`). Each agent
(:class:`samples.dark_pool.dark_pool_fcn_agent.DarkPoolFCNAgent`) decides its orders
from the lit market and sends each of them to the dark pool instead with the
probability ``darkPoolChance``. The dark pool executes its orders at the mid price of
the lit market. ``examples/dark_pool.ipynb`` plots and analyses the results.

Each printed line has the same columns as the output of plhamJ (see
:class:`samples.dark_pool.dark_pool_print_logger.DarkPoolPrintLogger`): session, time,
market ID, market name, market price, fundamental price, trade price and executed
volume.

The differences of this sample from plhamJ are as follows. ``examples/dark_pool.ipynb``
also lists the differences that come from PAMS itself, such as the lack of itayose at
the start of a session.

- ``config.json`` sets ``maxNormalOrders`` to ``2``, the number of markets, which is
  the default of plhamJ. The default of PAMS is ``1``.
- The dark orders are market orders, while plhamJ uses limit orders with the largest
  price. The dark pool rejects limit orders.
- plhamJ ignores the orders to the dark pool while the dark pool is not running. The
  dark pool of this sample places such orders and cancels them at once, so they are
  logged and counted as orders.
- plhamJ sends an order to the dark pool only if the mid price of the dark pool is not
  finite, which is always true because the dark pool has only orders without prices.
  This sample omits the condition.
- The trade price is ``nan`` instead of ``NaN`` when neither market executes orders.
"""

import argparse
import random
from typing import Optional

from pams.runners.sequential import SequentialRunner
from samples.dark_pool.dark_pool_fcn_agent import DarkPoolFCNAgent
from samples.dark_pool.dark_pool_market import DarkPoolMarket
from samples.dark_pool.dark_pool_print_logger import DarkPoolPrintLogger


def main() -> None:
    """Run the dark pool sample with the config and the seed in the arguments."""
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

    # the simulation needs a seedable PRNG, not a cryptographic one.
    runner = SequentialRunner(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,  # nosec B311
        logger=DarkPoolPrintLogger(),
    )
    runner.class_register(cls=DarkPoolMarket)
    runner.class_register(cls=DarkPoolFCNAgent)
    runner.main()


if __name__ == "__main__":
    main()
