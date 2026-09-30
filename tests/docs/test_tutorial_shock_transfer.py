import os
from pathlib import Path
from typing import Dict
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


def test_shock_transfer_uses_sample() -> None:
    # the page says that the config is the one of samples/shock_transfer
    tutorial = load_tutorial("tutorial_shock_transfer")
    assert tutorial.CONFIG == load_sample_config("shock_transfer")


def test_shock_transfer_short_run(capsys: pytest.CaptureFixture[str]) -> None:
    tutorial = load_tutorial("tutorial_shock_transfer")
    runner = tutorial.run_simulation(42, spot_weight=0.1, main_steps=20)
    # the run does not change CONFIG
    assert "fundamentalWeight" not in tutorial.CONFIG["FCNAgents-1"]
    assert tutorial.CONFIG["simulation"]["sessions"][1]["iterationSteps"] == 500
    assert runner.simulator.sessions[1].iteration_steps == 20

    spot1 = runner.simulator.name2market["SpotMarket-1"]
    spot2 = runner.simulator.name2market["SpotMarket-2"]
    index = runner.simulator.name2market["IndexMarket-I"]
    # the shock at step 100, and the index one step later, as the page says
    assert spot1.get_fundamental_price(99) == 300.0
    assert round(spot1.get_fundamental_price(100), 6) == 270.0
    assert round(spot1.get_fundamental_price(119), 6) == 270.0
    assert index.get_fundamental_price(100) == 300.0
    assert round(index.get_fundamental_price(101), 6) == 285.0
    assert set(spot2.get_fundamental_prices(times=range(120))) == {300.0}

    measures: Dict[str, float] = tutorial.measure(runner)
    assert list(measures) == ["premium", "spot 2", "arb spot 2", "arb index"]
    tutorial.print_prices(runner, times=[99, 100])
    lines: List[str] = capsys.readouterr().out.splitlines()
    assert lines[0] == text_blocks("shock_transfer.rst")[0][0]
    assert lines[1] == text_blocks("shock_transfer.rst")[0][1]


def test_shock_transfer_seed_42(tmp_path: Path) -> None:
    tutorial = load_tutorial("tutorial_shock_transfer")
    runner = tutorial.run_simulation(42)
    measures: Dict[str, float] = tutorial.measure(runner)
    # the page says: the arbitrage agents sell 26 shares of spot 2 and buy 18 shares
    # of the index, the index market price is below the index value on average, and
    # spot 2 stays below 300
    assert measures["arb spot 2"] == -26
    assert measures["arb index"] == 18
    assert measures["premium"] < 0
    assert measures["spot 2"] < 0
    spot2 = runner.simulator.name2market["SpotMarket-2"]
    assert min(spot2.get_market_prices(times=range(300, 500))) < 290
    path = str(tmp_path / "prices.png")
    tutorial.plot_prices(runner, path=path)
    assert os.path.getsize(path) > 0


def test_shock_transfer_comparison(capsys: pytest.CaptureFixture[str]) -> None:
    tutorial = load_tutorial("tutorial_shock_transfer")
    results = tutorial.compare([42], main_steps=20)
    assert list(results) == list(tutorial.CASES)
    for row in results.values():
        assert row["spot 2 up"] in (0, 1)
    tutorial.print_comparison(results, n_seeds=1)
    lines: List[str] = capsys.readouterr().out.splitlines()
    assert lines[0] == text_blocks("shock_transfer.rst")[1][0]
    assert [line[:19].strip() for line in lines[1:]] == list(tutorial.CASES)
    assert [line.split()[-3] for line in lines[1:]] == [
        f"{row['spot 2 up']:.0f}/1" for row in results.values()
    ]


def test_shock_transfer_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_tutorial("tutorial_shock_transfer")
    assert os.path.getsize(tmp_path / "shock_transfer_prices.png") > 0
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the prices of seed 42 and then the comparison
    blocks: List[List[str]] = text_blocks("shock_transfer.rst")
    assert len(blocks) == 2
    assert_shown(blocks[0] + blocks[1], lines)
    rows: Dict[str, List[str]] = {
        line[:19].strip(): line[19:].split() for line in lines[9:12]
    }
    # the page says: when spot 1 reacts faster, the premium is positive, the
    # arbitrage agents buy spot 2, and spot 2 is above 300 in 7 of the 10 seeds;
    # when the index reacts faster, the premium is negative, they buy the index and
    # sell spot 2, and spot 2 is the lowest
    assert float(rows["spot reacts faster"][0]) > 0
    assert float(rows["spot reacts faster"][3]) > 0
    assert rows["spot reacts faster"][2] == "7/10"
    assert float(rows["index reacts faster"][0]) < 0
    assert float(rows["index reacts faster"][3]) < 0
    assert float(rows["index reacts faster"][4]) > 0
    spot2: Dict[str, float] = {name: float(row[1]) for name, row in rows.items()}
    assert min(spot2, key=spot2.__getitem__) == "index reacts faster"
    assert max(spot2, key=spot2.__getitem__) == "spot reacts faster"
