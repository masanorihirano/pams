"""Dark pool sample.

This sample simulates a lit market and a dark pool beside it. It takes the dark pool
and the market selection without smart order routing (SOR) from the artificial market
model of Mizuta et al. (2015), https://doi.org/10.1007/s40844-015-0020-3. The lit
market is a continuous double auction. The dark pool
(:class:`samples.dark_pool.dark_pool_market.DarkPoolMarket`) accepts only market
orders, which have no price. A new order to it trades at once if an opposite order is
waiting there and waits otherwise, and the trade price is the mid price of the lit
market. Each agent (:class:`samples.dark_pool.dark_pool_fcn_agent.DarkPoolFCNAgent`)
decides its order to the lit market as :class:`pams.agents.FCNAgent` does. It then
sends the order to the dark pool with the probability ``darkPoolChance``, which is
``d`` of the paper, and to the lit market otherwise.

The agents differ from those of the paper in their order rule, which is that of
:class:`pams.agents.FCNAgent`, and in their parameters. For example, ``config.json``
sets the chart weight to 0, so the agents have no technical term. The Config section
of ``examples/dark_pool.ipynb`` compares the parameters, and the notebook also
explains the model and the measures of the paper and compares the results for several
values of ``darkPoolChance`` with the findings of the paper.

Each printed line shows one market at the end of one step: the session ID, the time,
the market ID, the market name, the market price, the fundamental price, the trade
price and the executed volume (see
:class:`samples.dark_pool.dark_pool_print_logger.DarkPoolPrintLogger`).

This sample is ported from the DarkPool sample of plhamJ and prints the same columns
as that sample. It differs from that sample as follows, and
``examples/dark_pool.ipynb`` also lists the differences that come from PAMS itself,
such as the lack of itayose at the start of a session.

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

    runner = SequentialRunner(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,
        logger=DarkPoolPrintLogger(),
    )
    runner.class_register(cls=DarkPoolMarket)
    runner.class_register(cls=DarkPoolFCNAgent)
    runner.main()


if __name__ == "__main__":
    main()
