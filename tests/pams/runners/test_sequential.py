import os.path
import random
import time
from typing import Dict
from typing import Type

from pams.runners import Runner
from pams.runners import SequentialRunner
from tests.pams.runners.test_base import TestRunner


class TestSequentialRunner(TestRunner):
    runner_class: Type[Runner] = SequentialRunner
    default_setting: Dict = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["FCNAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 10,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                }
            ],
        },
        "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
        "FCNAgents": {
            "class": "FCNAgent",
            "numAgents": 10,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": {"expon": [1.0]},
            "chartWeight": {"expon": [0.0]},
            "noiseWeight": {"expon": [1.0]},
            "meanReversionTime": {"uniform": [50, 100]},
            "noiseScale": 0.001,
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
        },
    }

    def test__(self) -> None:
        config = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "samples",
            "CI2002",
            "config.json",
        )
        runner = SequentialRunner(settings=config, prng=random.Random(42))
        runner._setup()
        runner.simulator._update_times_on_markets(markets=runner.simulator.markets)
        start_time = time.time()
        for _ in range(10000):
            _ = runner._collect_orders_from_normal_agents(
                session=runner.simulator.sessions[0]
            )
        end_time = time.time()
        time_per_step = (end_time - start_time) / 1000
        print("time/step", time_per_step)
        assert time_per_step < 0.003
