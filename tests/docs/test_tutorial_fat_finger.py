from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import matplotlib
import pytest

from pams.logs import OrderLog
from pams.runners import SequentialRunner
from tests.docs.sample_config import load_sample_config
from tests.docs.test_tutorials import assert_in_page
from tests.docs.test_tutorials import assert_same_image
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

    # the numbers in the prose
    wall: str = f"{mistake.price:.3f}"
    before: float = market.get_market_price(199)
    buy_199: Dict[Optional[float], int] = recorder.buy_books[199]
    buy_200: Dict[Optional[float], int] = recorder.buy_books[200]
    best_199: float = max(price for price in buy_199 if price is not None)
    best_200: float = max(price for price in buy_200 if price is not None)
    taken: int = mistake.volume - left[-1]
    # the buyers take about one share every two steps, from step 200 to step 599
    rate: float = taken / 400
    assert round(1 / rate) == 2
    assert market.get_market_price(400) < mistake.price
    assert_in_page(
        "fat_finger.rst",
        f"The market price at step 199 is {before:.3f}, so the mistaken order sells"
        f" at :math:`{before:.3f} \\times 0.95 = {wall}`.",
        f"At step 200 it meets every buy order at or above that price:"
        f" {sum(log.volume for log in at_200)} shares are executed, and the buy side"
        f" of the book falls from {sum(buy_199.values())} to {sum(buy_200.values())}"
        f" shares. The best buy price drops from {best_199:.3f} to {best_200:.3f}.",
        f"All {sum(log.volume for log in at_200)} shares trade at one price,"
        f" {at_200[0].price:.3f}:",
        f"The other {left[0]} shares stay in the book as a *wall*. From then on, any"
        f" buy order at or above {wall} is executed against the wall at the wall's"
        f" price, {wall}, so the market price can no longer rise above it;",
        "as at step 400. By step 599, the buyers have taken"
        f" {taken} of the {mistake.volume} shares.",
        "the price never exceeds the price of the mistaken order from step 201 on, and"
        f" ends the run {(300 - market.get_market_price(599)) / 3:.0f}% below the"
        " fundamental price.",
        f"at about one share every two steps, the {left[-1]} shares left would last"
        f" about {round(left[-1] / rate, -3):.0f} more steps, longer than the order's"
        " lifetime.",
        f"(here ``{runner.simulator.id2agent[mistake.agent_id].name}``)",
        f"Here it falls to {wall}, 5% below the market price of step 199,",
        f"``{owner.name}`` started with 50 shares and ends the run holding"
        f" {owner.asset_volumes[market.market_id]}.",
    )
    # the buy orders in the book before the mistake, in the plot of the book
    buy_prices: List[float] = [
        price
        for time in range(150, 200)
        for price in recorder.buy_books[time]
        if price is not None
    ]
    assert min(buy_prices) == pytest.approx(270, abs=1)
    assert_in_page(
        "fat_finger.rst",
        f"Until step 200 the buy orders lie between about 270 and"
        f" {max(buy_prices):.0f}.",
        f"Before step 200, buy orders sit up to about {max(buy_prices):.0f}.",
    )


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
    assert_in_page("fat_finger.rst", "With 100 shares, the wall is used up at step 347")


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
        assert_same_image(name, tmp_path / name)
    # the prose about the table of the volumes
    rows: Dict[str, List[str]] = {
        line.split()[0]: line.split() for line in blocks[2][1:]
    }
    assert rows["10000"][1] == "0/10"
    assert rows["100"][1] == "10/10"
    late: Dict[str, float] = {name: float(row[-1]) for name, row in rows.items()}
    assert_in_page(
        "fat_finger.rst",
        "the mean price of steps 500 to 599 is about"
        f" {(300 - late['10000']) / 3:.0f}% below the fundamental price.",
        "on average about"
        f" {round(float(rows['100'][2]) - 200, -1):.0f} steps after the mistake,",
        f"is only {late['none'] - late['100']:.1f} lower than without the mistake.",
        f"used up about {float(rows['10'][2]) - 200:.0f} steps after the mistake",
    )
