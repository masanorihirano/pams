from pathlib import Path
from typing import List

import matplotlib
import pytest

from tests.docs.sample_config import load_sample_config
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


def test_trading_halt_longer_halts() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    stats = tutorial.halt_stats(*tutorial.run_simulation(42, length=200))
    assert stats["starts"] == [127, 379]
    assert stats["halted"] == 2 * 201


def test_trading_halt_lower_threshold() -> None:
    tutorial = load_tutorial("tutorial_trading_halt")
    runner, recorder = tutorial.run_simulation(42, rate=0.02)
    stats = tutorial.halt_stats(runner, recorder)
    assert stats["halts"] == 5
    assert stats["starts"] == [111, 212, 313, 414, 528]
    # the last halt lasts until the end of the run
    assert recorder.halted[-1] == 599
    assert stats["last"] == stats["lowest"]


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
    halted_run, _ = tutorial.run_simulation(42)
    prices: List[float] = runner.simulator.name2market["Market"].get_market_prices()
    halted_prices: List[float] = halted_run.simulator.name2market[
        "Market"
    ].get_market_prices()
    assert prices[:128] == halted_prices[:128]
    assert prices != halted_prices


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
    run_tutorial("tutorial_trading_halt")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the text blocks of the page show the whole output, in order
    shown: List[str] = [
        line for block in text_blocks("trading_halt.rst") for line in block
    ]
    assert len(shown) == 10
    assert_shown(shown, lines)
    assert (tmp_path / "trading_halt_prices.png").stat().st_size > 0
