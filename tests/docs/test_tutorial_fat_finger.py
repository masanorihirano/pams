import os
from pathlib import Path
from typing import List

import matplotlib
import pytest

from pams.logs import OrderLog
from pams.runners import SequentialRunner
from tests.docs.sample_config import load_sample_config
from tests.docs.test_tutorials import TUTORIAL_DIR
from tests.docs.test_tutorials import assert_shown
from tests.docs.test_tutorials import load_tutorial
from tests.docs.test_tutorials import run_tutorial
from tests.docs.test_tutorials import text_blocks

# the tutorial imports matplotlib.pyplot: never open a window in the tests
matplotlib.use("Agg")

# the FCN agents place orders off the tick size, as the first tutorial explains
pytestmark = pytest.mark.filterwarnings(
    "ignore:order price does not accord to the tick size:UserWarning"
)


def test_fat_finger_uses_sample() -> None:
    tutorial = load_tutorial("tutorial_fat_finger")
    # the page says that the configuration is the one of samples/fat_finger
    assert tutorial.CONFIG == load_sample_config("fat_finger")


def test_fat_finger_seed_42() -> None:
    tutorial = load_tutorial("tutorial_fat_finger")
    runner, recorder = tutorial.run_simulation(42)
    market = runner.simulator.name2market["Market"]
    event = runner.simulator.name2event["OrderMistakeShock"]
    assert event.triggerd
    mistake: OrderLog = tutorial.find_mistake(runner, recorder)
    assert runner.simulator.id2agent[mistake.agent_id].name == "FCNAgents-28"
    assert (mistake.time, mistake.is_buy, mistake.volume) == (200, False, 10000)
    assert mistake.price == pytest.approx(market.get_market_price(199) * 0.95)
    # 18 shares are executed at step 200, all at the price of the last buy order
    at_200 = [log for log in recorder.executions if log.time == 200]
    assert sum(log.volume for log in at_200) == 18
    assert [log.price for log in at_200] == [pytest.approx(279.96282)] * len(at_200)
    assert all(log.sell_order_id == mistake.order_id for log in at_200)
    # the price never rises above the mistaken order after step 200
    assert max(market.get_market_prices(times=range(201, 600))) == mistake.price
    left: List[int] = [
        tutorial.wall_left(recorder, mistake, time) for time in range(200, 600)
    ]
    assert left[0] == 9982 and left[-1] == 9793
    assert left == sorted(left, reverse=True)
    # the owner sells shares that it does not have
    owner = runner.simulator.id2agent[mistake.agent_id]
    assert owner.asset_volumes[market.market_id] == -155


def test_fat_finger_without_mistake() -> None:
    tutorial = load_tutorial("tutorial_fat_finger")
    runner: SequentialRunner = tutorial.run_simulation(42, enabled=False)[0]
    assert "OrderMistakeShock" not in runner.simulator.name2event
    first = load_tutorial("tutorial_first_simulation")
    first_runner: SequentialRunner = first.run_simulation(42)[0]
    times = range(0, 600)
    assert runner.simulator.name2market["Market"].get_market_prices(
        times=times
    ) == first_runner.simulator.name2market["Market"].get_market_prices(times=times)


def test_fat_finger_small_mistake() -> None:
    tutorial = load_tutorial("tutorial_fat_finger")
    # the page says that 100 shares are used up at step 347
    runner, recorder = tutorial.run_simulation(42, volume=100)
    mistake: OrderLog = tutorial.find_mistake(runner, recorder)
    assert mistake.volume == 100
    assert tutorial.wall_left(recorder, mistake, 346) > 0
    assert tutorial.wall_left(recorder, mistake, 347) == 0


def test_fat_finger_compare_volumes() -> None:
    tutorial = load_tutorial("tutorial_fat_finger")
    results = tutorial.compare_volumes(volumes=(10, None), seeds=[42, 43])
    assert list(results) == ["10", "none"]
    for used_up, _ in results["10"]:
        assert used_up is not None and used_up >= 200
    assert [used_up for used_up, _ in results["none"]] == [None, None]


def test_fat_finger_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_tutorial("tutorial_fat_finger")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the whole output, in three blocks
    blocks: List[List[str]] = text_blocks("fat_finger.rst")
    assert len(blocks) == 3
    assert_shown([line for block in blocks for line in block], lines)
    for name in ["fat_finger_prices.png", "fat_finger_book.png"]:
        assert (tmp_path / name).is_file()
        assert os.path.isfile(os.path.join(TUTORIAL_DIR, "images", name))
