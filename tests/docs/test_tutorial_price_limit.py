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
    # until step 147, when the price first reaches a limit
    tutorial = load_tutorial("tutorial_price_limit")
    with_limit: List[float] = (
        tutorial.run_simulation(42).simulator.name2market["Market"].get_market_prices()
    )
    without_limit: List[float] = (
        tutorial.run_simulation(42, enabled=False)
        .simulator.name2market["Market"]
        .get_market_prices()
    )
    assert with_limit[:147] == without_limit[:147]
    assert with_limit[147] != without_limit[147]
    assert with_limit[147] == pytest.approx(285.0)


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
    run_tutorial("tutorial_price_limit")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the text blocks of the page show the whole output, in order
    shown: List[str] = [
        line for block in text_blocks("price_limit.rst") for line in block
    ]
    assert len(shown) == 12
    assert_shown(shown, lines)
    assert (tmp_path / "price_limit_prices.png").stat().st_size > 0
