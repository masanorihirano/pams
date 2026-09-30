import statistics
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib
import pytest

from pams.runners import SequentialRunner
from tests.docs.sample_config import load_sample_config
from tests.docs.test_tutorials import assert_in_page
from tests.docs.test_tutorials import assert_same_image
from tests.docs.test_tutorials import assert_shown
from tests.docs.test_tutorials import load_tutorial
from tests.docs.test_tutorials import record_runners
from tests.docs.test_tutorials import run_tutorial
from tests.docs.test_tutorials import text_blocks

# the tutorial imports matplotlib.pyplot: never open a window in the tests
matplotlib.use("Agg")

# the FCN agents place orders off the tick size, as the first tutorial explains
pytestmark = pytest.mark.filterwarnings(
    "ignore:order price does not accord to the tick size:UserWarning"
)


def main_prices(runner: SequentialRunner) -> List[float]:
    """Read the market prices of the main session, steps 100 to 599."""
    return runner.simulator.name2market["Market"].get_market_prices(
        times=range(100, 600)
    )


def limit_stays(prices: List[float], tick: float) -> List[Tuple[int, int, str]]:
    """Find the first and last steps of each stay at one limit, and the limit."""
    stays: List[Tuple[int, int, str]] = []
    for time, price in enumerate(prices, start=100):
        if price >= 315.0 - tick:
            side = "upper"
        elif price <= 285.0 + tick:
            side = "lower"
        else:
            continue
        if len(stays) > 0 and stays[-1][1] == time - 1 and stays[-1][2] == side:
            stays[-1] = (stays[-1][0], time, side)
        else:
            stays.append((time, time, side))
    return stays


def test_limit_stays() -> None:
    prices: List[float] = [300.0, 315.0, 315.0, 285.0, 300.0, 315.0]
    assert limit_stays(prices, 0.00001) == [
        (101, 102, "upper"),
        (103, 103, "lower"),
        (105, 105, "upper"),
    ]


def test_price_limit_uses_sample() -> None:
    # the page says that the config is the one of samples/price_limit
    tutorial = load_tutorial("tutorial_price_limit")
    assert tutorial.CONFIG == load_sample_config("price_limit")


def test_price_limit_keeps_prices_in_band() -> None:
    tutorial = load_tutorial("tutorial_price_limit")
    runner = tutorial.run_simulation(42)
    assert runner.simulator.sessions[1].iteration_steps == 500
    stats = tutorial.limit_stats(runner)
    assert stats["lowest"] >= 285.0 - tutorial.TICK
    assert stats["highest"] <= 315.0 + tutorial.TICK
    assert stats["upper"] > 0
    assert stats["lower"] > 0
    # the price goes twice from one limit straight to the other, so it arrives at a
    # limit 32 times in 30 stays at either limit
    stays = limit_stays(main_prices(runner), tutorial.TICK)
    upper: int = sum(1 for _, _, side in stays if side == "upper")
    lower: int = sum(1 for _, _, side in stays if side == "lower")
    apart: int = 1 + sum(1 for a, b in zip(stays, stays[1:]) if b[0] > a[1] + 1)
    assert (len(stays), apart) == (32, 30)
    assert_in_page(
        "price_limit.rst",
        f"It arrives at a limit {len(stays)} times ({upper} times at 315 and {lower}"
        f" times at 285), and its longest stay is {stats['longest']} steps.",
    )
    assert (
        stats["changed"]
        == runner.simulator.name2event["PriceLimitRule"].activation_count
    )
    assert stats["changed"] > 0


def test_price_limit_disabled() -> None:
    tutorial = load_tutorial("tutorial_price_limit")
    runner = tutorial.run_simulation(42, enabled=False)
    # a disabled event is not in name2event, as the page says
    assert "PriceLimitRule" not in runner.simulator.name2event
    stats = tutorial.limit_stats(runner)
    assert stats["changed"] == 0
    assert stats["highest"] > 315.0


def test_price_limit_same_prices_until_first_limit() -> None:
    # the page says that the runs with and without the limit have the same prices
    # until the price first reaches a limit
    tutorial = load_tutorial("tutorial_price_limit")
    with_limit: List[float] = (
        tutorial.run_simulation(42).simulator.name2market["Market"].get_market_prices()
    )
    without_limit: List[float] = (
        tutorial.run_simulation(42, enabled=False)
        .simulator.name2market["Market"]
        .get_market_prices()
    )
    first: int = next(
        time
        for time, (price, other) in enumerate(zip(with_limit, without_limit))
        if price != other
    )
    assert with_limit[first] == pytest.approx(285.0)
    assert_in_page(
        "price_limit.rst",
        f"Their prices are the same until step {first}, when the price first reaches"
        " a limit",
    )


def test_price_limit_runaway() -> None:
    # without the limit, strong trend followers make the price run away
    tutorial = load_tutorial("tutorial_price_limit")
    with pytest.raises(OverflowError):
        tutorial.run_simulation(42, enabled=False, chart_weight=1.0)
    for seed in [43, 44]:
        runner = tutorial.run_simulation(seed, enabled=False, chart_weight=1.0)
        assert tutorial.limit_stats(runner)["lowest"] == pytest.approx(tutorial.TICK)
    assert_in_page(
        "price_limit.rst",
        "for seed 42 the run stops with an ``OverflowError``, and for the seeds 43 and"
        " 44 the price falls to 0.00001, the tick size.",
    )


def test_price_limit_compare() -> None:
    tutorial = load_tutorial("tutorial_price_limit")
    results = tutorial.compare(seeds=[42, 43], chart_weights=[0.0, 1.0], main_steps=50)
    assert list(results) == [0.0, 1.0]
    for runs in results.values():
        assert len(runs) == 2
        for stats in runs:
            assert stats["upper"] + stats["lower"] <= 50
            assert stats["longest"] <= 50


def test_price_limit_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    runners: List[SequentialRunner] = record_runners(monkeypatch)
    run_tutorial("tutorial_price_limit")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the text blocks of the page show the whole output, in order
    shown: List[str] = [
        line for block in text_blocks("price_limit.rst") for line in block
    ]
    assert len(shown) == 12
    assert_shown(shown, lines)
    assert_same_image("price_limit_prices.png", tmp_path / "price_limit_prices.png")
    tutorial = load_tutorial("tutorial_price_limit")
    assert len(runners) == 33
    assert {runner.simulator.sessions[1].iteration_steps for runner in runners} == {500}
    assert_in_page("price_limit.rst", "It runs 33 simulations of 600 steps.")

    # the three runs of seed 42, in the order of CASES
    cases: Dict[str, Dict[str, float]] = {
        name: tutorial.limit_stats(runner)
        for name, runner in zip(tutorial.CASES, runners)
    }
    limit, no_limit = cases["limit"], cases["no limit"]
    assert_in_page(
        "price_limit.rst",
        f"Without the rule, the price moves between {no_limit['lowest']:.2f} and"
        f" {no_limit['highest']:.2f}. It is at or above 315 in {no_limit['upper']} of"
        f" the 500 steps, and at or below 285 in {no_limit['lower']} steps. With the"
        f" rule, the price never leaves the band: it is at 315 in {limit['upper']}"
        f" steps and at 285 in {limit['lower']} steps, and the rule changed the prices"
        f" of {limit['changed']} orders.",
        f"the price is at a limit in {limit['upper'] + limit['lower']} steps, against"
        f" {no_limit['upper'] + no_limit['lower']} steps at or beyond one without the"
        " rule.",
    )
    # the lower panel of the figure
    stays = limit_stays(main_prices(runners[2]), tutorial.TICK)
    upper = [stay for stay in stays if stay[2] == "upper"]
    lower = [stay for stay in stays if stay[2] == "lower"]
    assert len(upper) == 3 and upper[-1][0] == upper[-1][1]
    assert stays[-1][1] == 599
    assert_in_page(
        "price_limit.rst",
        f"It reaches 315 at step {upper[0][0]} and stays there until step"
        f" {upper[0][1]}, then comes back to 315 twice, the last time at step"
        f" {upper[-1][0]}. It reaches 285 at step {lower[0][0]}, and the last"
        f" {stays[-1][1] - stays[-1][0] + 1} steps are all at 285:",
    )

    # the ten seeds of each chart weight, in the order of compare()
    results: Dict[float, List[Dict[str, float]]] = {
        chart_weight: [
            tutorial.limit_stats(runner) for runner in runners[3 + 10 * i : 13 + 10 * i]
        ]
        for i, chart_weight in enumerate(tutorial.CHART_WEIGHTS)
    }
    longest: Dict[float, List[float]] = {
        chart_weight: [run["longest"] for run in runs]
        for chart_weight, runs in results.items()
    }
    at_limit: Dict[float, float] = {
        chart_weight: statistics.mean(run["upper"] + run["lower"] for run in runs)
        for chart_weight, runs in results.items()
    }
    # the seeds of a chart weight of 1.0 whose longest stay lasts until the end
    until_end: List[int] = [
        seed
        for seed, runner, run in zip(tutorial.SEEDS, runners[23:], results[1.0])
        if all(
            price >= 315.0 - tutorial.TICK or price <= 285.0 + tutorial.TICK
            for price in main_prices(runner)[-int(run["longest"]) :]
        )
    ]
    assert len(until_end) == 3
    assert sum(1 for stay in longest[1.0] if stay <= 16) == 6
    top: List[Tuple[float, int]] = sorted(
        zip(longest[1.0], tutorial.SEEDS), reverse=True
    )
    assert_in_page(
        "price_limit.rst",
        f"the longest stay is longer on average: {statistics.mean(longest[1.0]):.1f}"
        " steps with a chart weight of 1.0 against"
        f" {statistics.mean(longest[0.0]):.1f} steps without trend followers.",
        "With a chart weight of 1.0, six"
        " of the ten seeds have a longest stay of 16 steps or fewer",
        f"The mean comes mostly from the seeds {top[0][1]} ({top[0][0]} steps) and"
        f" {top[1][1]} ({top[1][0]} steps).",
        f"Three of the stays (seeds {until_end[0]}, {until_end[1]} and {until_end[2]})"
        " last until the end of the run",
        "The price spends fewer steps at a limit in total with trend followers"
        f" ({at_limit[1.0]:.1f} against {at_limit[0.0]:.1f}), but in longer stays.",
        "without trend followers the longest stay in the ten seeds is"
        f" {max(longest[0.0])} steps.",
    )
