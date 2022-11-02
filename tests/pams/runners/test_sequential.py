import os.path
import random
import time

from pams.runners import SequentialRunner


class TestSequentialRunner:
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
            orders = runner._collect_orders(session=runner.simulator.sessions[0])
        end_time = time.time()
        time_per_step = (end_time - start_time) / 1000
        print("time/step", time_per_step)
        assert time_per_step < 0.002
