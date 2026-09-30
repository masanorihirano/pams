import os
import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import matplotlib
import pytest

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


def test_market_share_uses_samples() -> None:
    tutorial = load_tutorial("tutorial_market_share")
    # the page says that the configurations are those of samples/market_share
    assert tutorial.CONFIG_TICK == load_sample_config("market_share")
    assert tutorial.CONFIG_MM == load_sample_config("market_share", "config-mm.json")


def test_market_share_config_changes() -> None:
    tutorial = load_tutorial("tutorial_market_share")
    config: Dict[str, Any] = tutorial.with_tick_size(2.0)
    assert config["Market-A"]["tickSize"] == 2.0
    assert tutorial.CONFIG_TICK["Market-A"]["tickSize"] == 10.0
    assert "MarketMakerAgent" not in tutorial.with_spread(None)["simulation"]["agents"]
    config = tutorial.with_spread(0.01)
    assert config["MarketMakerAgent"]["netInterestSpread"] == 0.01
    assert tutorial.CONFIG_MM["MarketMakerAgent"]["netInterestSpread"] == 0.02


def test_market_share_extended_market() -> None:
    tutorial = load_tutorial("tutorial_market_share")
    runner: SequentialRunner = tutorial.run_simulation(
        tutorial.CONFIG_TICK, seed=42, main_steps=10
    )
    volumes: List[int] = [
        runner.simulator.name2market[name].get_executed_volumes(times=[0])[0]
        for name in ["Market-A", "Market-B"]
    ]
    assert volumes == [90, 10]
    assert runner.simulator.sessions[1].iteration_steps == 10
    # without tradeVolume, the market starts with no volume like pams.Market
    config: Dict[str, Any] = tutorial.with_tick_size(10.0)
    del config["Market-B"]["tradeVolume"]
    runner = tutorial.run_simulation(config, seed=42, main_steps=10)
    market_b = runner.simulator.name2market["Market-B"]
    assert market_b.get_executed_volumes(times=[0]) == [0]
    # tradeVolume must be an int
    config = tutorial.with_tick_size(10.0)
    config["Market-B"]["tradeVolume"] = 10.0
    runner = SequentialRunner(settings=config, prng=random.Random(42))
    runner.class_register(cls=tutorial.ExtendedMarket)
    with pytest.raises(ValueError, match="tradeVolume must be int"):
        runner.main()


def test_market_share_tick_seed_42() -> None:
    tutorial = load_tutorial("tutorial_market_share")
    # the page says that Market-A trades only at 300 and 310 and stops after step 834
    runner: SequentialRunner = tutorial.run_simulation(
        tutorial.with_tick_size(10.0), seed=42, main_steps=900
    )
    market_a = runner.simulator.name2market["Market-A"]
    volumes: List[int] = market_a.get_executed_volumes(times=range(100, 1000))
    assert max(time for time, volume in enumerate(volumes, 100) if volume > 0) == 834
    assert set(market_a.get_market_prices(times=range(100, 1000))) == {300.0, 310.0}
    shares: List[float] = tutorial.share_of_b(runner)
    assert len(shares) == 9 and shares[-1] == 1.0


def test_market_share_mm_seed_42() -> None:
    tutorial = load_tutorial("tutorial_market_share")
    # the page says that Market-B has no trade in the second session with seed 42
    for spread in [0.02, 0.01, 0.0001]:
        runner: SequentialRunner = tutorial.run_simulation(
            tutorial.with_spread(spread), seed=42, main_steps=500
        )
        market_b = runner.simulator.name2market["Market-B"]
        assert sum(market_b.get_executed_volumes(times=range(100, 600))) == 0


def test_market_share_compare() -> None:
    tutorial = load_tutorial("tutorial_market_share")
    results = tutorial.compare(
        {"none": tutorial.with_spread(None), "0.01": tutorial.with_spread(0.01)},
        seeds=[43, 44],
        main_steps=200,
    )
    assert list(results) == ["none", "0.01"]
    for runs in results.values():
        assert len(runs) == 2
        for shares in runs:
            assert len(shares) == 2 and all(0.0 <= share <= 1.0 for share in shares)


def test_market_share_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_tutorial("tutorial_market_share")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the whole output, in three blocks
    blocks: List[List[str]] = text_blocks("market_share.rst")
    assert len(blocks) == 3
    assert_shown([line for block in blocks for line in block], lines)
    for name in ["market_share_tick.png", "market_share_mm.png"]:
        assert (tmp_path / name).is_file()
        assert os.path.isfile(os.path.join(TUTORIAL_DIR, "images", name))
