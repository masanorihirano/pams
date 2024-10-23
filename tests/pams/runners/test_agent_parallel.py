import copy
import time
from typing import Dict
from typing import Type
from typing import cast

import pytest

from pams.runners import MultiProcessAgentParallelRuner
from pams.runners import MultiThreadAgentParallelRuner
from pams.runners import Runner
from pams.runners.sequential import SequentialRunner
from tests.pams.runners.test_base import TestRunner

from .delay_agent import FCNDelayAgent
from .delay_agent import wait_time


class TestMultiThreadAgentParallelRuner(TestRunner):
    runner_class: Type[Runner] = MultiThreadAgentParallelRuner
    default_setting: Dict = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["FCNAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 5,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                    "events": ["FundamentalPriceShock"],
                    "maxNormalOrders": 3,
                }
            ],
            "numParallel": 3,
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
        "FundamentalPriceShock": {
            "class": "FundamentalPriceShock",
            "target": "Market",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": 1,
            "enabled": True,
        },
    }

    def test_parallel_efficiency(self) -> None:

        setting = copy.deepcopy(self.default_setting)
        setting["FCNAgents"]["class"] = "FCNDelayAgent"  # Use the delayed agent

        runner_class_dummy = self.runner_class
        self.runner_class = SequentialRunner  # Temporarily set to SequentialRunner
        sequential_runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        self.runner_class = runner_class_dummy
        parallel_runner = cast(
            self.runner_class,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )

        sequential_runner.class_register(cls=FCNDelayAgent)
        parallel_runner.class_register(cls=FCNDelayAgent)
        start_time = time.time()
        sequential_runner.main()
        end_time = time.time()
        elps_time_sequential = end_time - start_time
        start_time = time.time()
        parallel_runner.main()
        end_time = time.time()
        elps_time_parallel = end_time - start_time
        assert elps_time_sequential < wait_time * 15 + 1
        assert elps_time_sequential > wait_time * 15
        assert elps_time_parallel < wait_time * 5 + 1
        assert elps_time_parallel > wait_time * 5

    def test_parallel_thread_warning(self) -> None:
        settings = copy.deepcopy(self.default_setting)
        settings["simulation"]["numParallel"] = 5
        runner = self.test__init__(
            setting_mode="dict", logger=None, simulator_class=None, setting=settings
        )
        with pytest.warns(UserWarning, match="is set to a larger value."):
            runner.main()


class TestMultiProcessAgentParallelRuner(TestMultiThreadAgentParallelRuner):
    runner_class: Type[Runner] = MultiProcessAgentParallelRuner
