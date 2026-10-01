import statistics
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

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


def below(price: float) -> float:
    """Compute how far a price is below 300, in percent."""
    return (300.0 - price) / 300.0 * 100


def test_trading_halt_uses_sample() -> None:
    # the page says that the config is the one of samples/trading_halt
    tutorial = load_tutorial("tutorial_trading_halt")
    assert tutorial.CONFIG == load_sample_config("trading_halt")


def test_trading_halt_starts() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    assert tutorial.halt_starts([], 2) == []
    # a halt that starts at step 5 covers the ends of the steps 5 to 7
    assert tutorial.halt_starts([5, 6, 7, 20, 21, 22], 2) == [5, 20]
    # a new halt can start in the step in which the previous one ends
    assert tutorial.halt_starts([5, 6, 7, 8, 9, 10], 2) == [5, 8]


def test_trading_halt_halts() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    runner, recorder = tutorial.run_simulation(42)
    market = runner.simulator.name2market["Market"]
    # the shock lowers the fundamental price by 10% at the first step of the session
    assert market.get_fundamental_price(99) == pytest.approx(300.0)
    assert market.get_fundamental_price(100) == pytest.approx(270.0)
    stats = tutorial.halt_stats(runner, recorder)
    assert stats["halts"] == 2
    assert stats["starts"] == [127, 275]
    # each halt covers 101 step ends
    assert stats["halted"] == 2 * 101
    assert recorder.halted == list(range(127, 228)) + list(range(275, 376))
    # the first halt comes at a fall of 5% and the second at a fall of 10%
    assert market.get_market_price(127) <= 300.0 * 0.95
    assert market.get_market_price(275) <= 300.0 * 0.90
    # the price does not change during a halt
    assert set(market.get_market_prices(times=range(127, 228))) == {
        market.get_market_price(127)
    }
    # the orders placed during the first halt trade when the market reopens
    assert market.get_executed_volume(228) == 27
    first, second = stats["starts"]
    first_price: float = market.get_market_price(first)
    second_price: float = market.get_market_price(second)
    # a third halt would need a fall of 15%
    assert stats["lowest"] > 255.0
    assert_in_page(
        "trading_halt.rst",
        f"At step {first}, a trade at {first_price:.2f}, {below(first_price):.1f}%"
        " below 300, starts the first halt.",
        f"The market trades again from step {first + 101}. The orders placed during"
        f" the halt trade at once: {market.get_executed_volume(first + 101)} shares"
        " change hands in this step, and the price moves to"
        f" {market.get_market_price(first + 101):.2f}.",
        f"At step {second}, a trade at {second_price:.2f},"
        f" {below(second_price):.1f}% below 300, starts the second halt. After it,"
        f" {market.get_executed_volume(second + 101)} shares trade at once at step"
        f" {second + 101}. A third halt would need a fall of 15%, to 255, which does"
        " not happen.",
        f"The market is halted at the end of {stats['halted']} of the 500 steps: two"
        " halts of 101 steps.",
    )


def test_trading_halt_longer_halts() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    stats = tutorial.halt_stats(*tutorial.run_simulation(42, length=200))
    assert stats["starts"] == [127, 379]
    assert stats["halted"] == 2 * 201
    no_halt = tutorial.halt_stats(*tutorial.run_simulation(42, enabled=False))
    assert_in_page(
        "trading_halt.rst",
        "the halts start at steps"
        f" {stats['starts'][0]} and {stats['starts'][1]}, and the market is halted in"
        f" {stats['halted']} of the 500 steps. The lowest price is"
        f" {stats['lowest']:.2f}, closer to 270 than in the first two cases, but only"
        f" {stats['traded']} shares trade, against {no_halt['traded']} without halts.",
    )


def test_trading_halt_lower_threshold() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    runner, recorder = tutorial.run_simulation(42, rate=0.02)
    stats = tutorial.halt_stats(runner, recorder)
    assert stats["halts"] == 5
    assert stats["starts"] == [111, 212, 313, 414, 528]
    # the last halt lasts until the end of the run
    assert recorder.halted[-1] == 599
    assert stats["last"] == stats["lowest"]
    # the n-th halt comes at a fall of 2n%
    market = runner.simulator.name2market["Market"]
    before: List[float] = [market.get_market_price(t - 1) for t in stats["starts"]]
    after: List[float] = [market.get_market_price(t) for t in stats["starts"]]
    thresholds: List[float] = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert all(below(price) >= limit for price, limit in zip(after, thresholds))
    # at step 212, the jump moves the price past the threshold; at the steps 313 and
    # 414, the price is past the threshold before the market trades again
    assert below(before[1]) < thresholds[1]
    assert below(before[2]) >= thresholds[2] and below(before[3]) >= thresholds[3]
    assert before[2] == after[1] and before[3] == after[2]
    assert_in_page(
        "trading_halt.rst",
        "the market halts five times, at steps 111, 212, 313, 414 and 528.",
        f"This jump moves the price from {before[1]:.2f} ({below(before[1]):.1f}%"
        f" below 300) to {after[1]:.2f} ({below(after[1]):.1f}% below), past the"
        " threshold of 4%.",
        "At steps 313 and 414, the price is already past the next threshold when the"
        f" market trades again: {before[2]:.2f} is past 6%, and {before[3]:.2f}, the"
        " price from step 313, is past 8%.",
        f"The market is halted at the end of {stats['halted']} of the 500 steps, and"
        " the price falls in steps.",
        f"the last price is also the lowest, {stats['lowest']:.2f}.",
    )


def test_trading_halt_disabled() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    runner, recorder = tutorial.run_simulation(42, enabled=False)
    # a disabled event is not in name2event, as the page says
    assert "TradingHaltRule" not in runner.simulator.name2event
    stats = tutorial.halt_stats(runner, recorder)
    assert stats["halts"] == 0
    assert stats["halted"] == 0
    assert stats["starts"] == []
    # the page says that the runs with and without halts are the same until the
    # first halt at step 127
    halted_run, halted_recorder = tutorial.run_simulation(42)
    prices: List[float] = runner.simulator.name2market["Market"].get_market_prices()
    halted_prices: List[float] = halted_run.simulator.name2market[
        "Market"
    ].get_market_prices()
    assert prices[:128] == halted_prices[:128]
    assert prices != halted_prices
    lowest: float = min(prices[100:])
    halted_stats = tutorial.halt_stats(halted_run, halted_recorder)
    assert_in_page(
        "trading_halt.rst",
        "Without halts (the second row), the run is the same until step 127. The price"
        f" then falls below the new fundamental price, to {lowest:.2f} at step"
        f" {prices.index(lowest)}, and ends at {prices[-1]:.2f}. In this run the halts"
        f" hardly change the lowest price ({halted_stats['lowest']:.2f} against"
        f" {lowest:.2f}).",
    )


def test_trading_halt_compare() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    results = tutorial.compare(seeds=[42, 43], main_steps=50)
    assert list(results) == list(tutorial.CASES)
    for runs in results.values():
        assert len(runs) == 2
        for stats in runs:
            assert stats["halted"] <= 50
            # every halt starts in the main session
            assert len(stats["starts"]) == stats["halts"]


def test_trading_halt_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    runners: List[SequentialRunner] = record_runners(monkeypatch)
    run_tutorial("tutorial_trading_halt")
    tutorial = load_tutorial("tutorial_trading_halt")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the text blocks of the page show the whole output, in order
    shown: List[str] = [
        line for block in text_blocks("trading_halt.rst") for line in block
    ]
    assert len(shown) == 10
    assert_shown(shown, lines)
    assert_same_image("trading_halt_prices.png", tmp_path / "trading_halt_prices.png")
    assert len(runners) == 44
    assert {runner.simulator.sessions[1].iteration_steps for runner in runners} == {500}
    assert_in_page("trading_halt.rst", "It runs 44 simulations of 600 steps.")

    # the runs of compare(), by case and then by seed
    results: Dict[str, List[Dict[str, Any]]] = {
        name: [
            tutorial.halt_stats(runner, runner.logger)
            for runner in runners[4 + 10 * i : 14 + 10 * i]
        ]
        for i, name in enumerate(tutorial.CASES)
    }
    halt_5, no_halt, halt_200, halt_2 = results.values()
    assert {run["halts"] for run in halt_5} == {2}
    assert {run["halts"] for run in halt_200} == {2}
    assert {run["halts"] for run in halt_2} == {5}
    no_halt_lowest: Dict[int, float] = {
        seed: run["lowest"] for seed, run in zip(tutorial.SEEDS, no_halt)
    }
    low_seeds: List[int] = sorted(
        sorted(no_halt_lowest, key=no_halt_lowest.__getitem__)[:2]
    )
    others: List[float] = [
        lowest for seed, lowest in no_halt_lowest.items() if seed not in low_seeds
    ]
    assert 255.0 <= min(others) and max(others) <= 260.0
    assert 255.0 <= min(run["lowest"] for run in halt_5)
    assert max(run["lowest"] for run in halt_5) <= 262.0
    halted: List[float] = [
        statistics.mean(run["halted"] for run in runs) / 500 * 100
        for runs in [halt_5, halt_200, halt_2]
    ]
    assert_in_page(
        "trading_halt.rst",
        "Every seed has two halts at 5% and five at 2%.",
        "the mean lowest price is"
        f" {statistics.mean(run['lowest'] for run in no_halt):.2f} without halts,"
        f" {statistics.mean(run['lowest'] for run in halt_5):.2f} with halts of 100"
        f" steps, {statistics.mean(run['lowest'] for run in halt_200):.2f} with halts"
        f" of 200 steps and {statistics.mean(run['lowest'] for run in halt_2):.2f} with"
        " the 2% threshold.",
        "the mean last price is"
        f" {statistics.mean(run['last'] for run in halt_5):.2f} with halts of 100"
        f" steps, against {statistics.mean(run['last'] for run in no_halt):.2f}"
        " without halts.",
        "Without halts, the lowest price is"
        f" {no_halt_lowest[low_seeds[0]]:.1f} for seed {low_seeds[0]} and"
        f" {no_halt_lowest[low_seeds[1]]:.1f} for seed {low_seeds[1]}. For the other"
        " eight seeds, it is between 255 and 260, close to the lowest prices with"
        " halts of 100 steps (255 to 262).",
        f"The market is halted in about {halted[0]:.0f}%, {halted[1]:.0f}% and"
        f" {halted[2]:.0f}% of the second session",
    )
