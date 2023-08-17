import os.path
import random
from unittest import mock

import pytest

from pams import Simulator
from pams.logs import Logger
from samples.market_share.main import ExtendedMarket
from samples.market_share.main import main
from tests.pams.test_market import TestMarket


class TestExtendedMarket(TestMarket):
    base_class = ExtendedMarket

    def test_setup(self) -> None:
        m = self.base_class(
            market_id=0,
            prng=random.Random(42),
            logger=Logger(),
            simulator=Simulator(prng=random.Random(42)),
            name="test",
        )
        m.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
                "tradeVolume": 100,
            }
        )
        with pytest.raises(ValueError):
            m.setup(
                settings={
                    "tickSize": 0.001,
                    "outstandingShares": 100,
                    "marketPrice": 300.0,
                    "fundamentalPrice": 500.0,
                    "tradeVolume": "error",
                }
            )


def test_market_share() -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    sample_dir = os.path.join(root_dir, "samples", "market_share")
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{sample_dir}/config.json", "--seed", "1"]
    ):
        main()
    with mock.patch(
        "sys.argv",
        ["main.py", "--config", f"{sample_dir}/config-mm.json", "--seed", "1"],
    ):
        main()
