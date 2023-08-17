import argparse
import random
from typing import Any
from typing import Dict
from typing import Optional

from pams import Market
from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner


class ExtendedMarket(Market):
    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:
        super(ExtendedMarket, self).setup(settings=settings, *args, **kwargs)
        if "tradeVolume" in settings:
            if not isinstance(settings["tradeVolume"], int):
                raise ValueError("tradeVolume must be int")
            self._executed_volumes = [int(settings["tradeVolume"])]


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
    runner.class_register(cls=ExtendedMarket)
    runner.main()


if __name__ == "__main__":
    main()
