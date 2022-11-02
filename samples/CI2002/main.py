import argparse
import random
from typing import Optional

from pams import Logger
from pams.logs import MarketStepEndLog
from pams.runners.sequential import SequentialRunner


class PrintLogger(Logger):
    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        print(
            f"{log.session.session_id} {log.market.get_time()} {log.market.market_id} {log.market.name} {log.market.get_market_price()} {log.market.get_fundamental_price()}"
        )


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
        logger=PrintLogger(),
    )
    runner.main()


if __name__ == "__main__":
    main()
