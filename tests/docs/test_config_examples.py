import glob
import os
import random
import warnings
from typing import List
from typing import Type

import pytest

from pams.logs import MarketStepSaver
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners import SequentialRunner

EXAMPLE_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "docs",
    "source",
    "user_guide",
    "config_examples",
)
EXAMPLES: List[str] = sorted(glob.glob(os.path.join(EXAMPLE_DIR, "*.json")))


def test_examples_exist() -> None:
    assert len(EXAMPLES) > 0


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_config_example(path: str) -> None:
    runner_class: Type[SequentialRunner] = (
        MultiThreadAgentParallelRunner
        if os.path.basename(path) == "parallel.json"
        else SequentialRunner
    )
    saver = MarketStepSaver()
    with open(path, encoding="utf-8") as fp:
        runner = runner_class(settings=fp, prng=random.Random(42), logger=saver)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.main()
    messages: List[str] = [str(w.message) for w in caught]
    # examples in the docs must not use deprecated or obsolete keys
    assert not any("replaced" in m or "obsolete" in m for m in messages)
    # the parallel example must not be limited by maxNormalOrders
    assert not any("limited by max_normal_orders" in m for m in messages)
    n_steps: int = sum(session.iteration_steps for session in runner.simulator.sessions)
    assert len(saver.market_step_logs) == n_steps * len(runner.simulator.markets)


def test_events_example_timing() -> None:
    # "triggerTime": 100 in the session starting at step 100 -> shock at step 200
    saver = MarketStepSaver()
    with open(os.path.join(EXAMPLE_DIR, "events.json"), encoding="utf-8") as fp:
        runner = SequentialRunner(settings=fp, prng=random.Random(42), logger=saver)
    runner.main()
    fundamental = {
        log["market_time"]: log["fundamental_price"] for log in saver.market_step_logs
    }
    assert fundamental[199] == pytest.approx(300.0)
    assert fundamental[200] == pytest.approx(270.0)
