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


def test_ci2002_uses_sample() -> None:
    # the page says that the config is the one of samples/CI2002
    tutorial = load_tutorial("tutorial_ci2002")
    assert tutorial.CONFIG == load_sample_config("CI2002")


def test_ci2002_cases() -> None:
    tutorial = load_tutorial("tutorial_ci2002")
    config = tutorial.make_config(tutorial.CASES["more chart"])
    assert config["FCNAgents"]["chartWeight"] == {"expon": [0.5]}
    assert config["FCNAgents"]["fundamentalWeight"] == {"expon": [1.0]}
    # CONFIG itself is not changed
    assert tutorial.CONFIG["FCNAgents"]["chartWeight"] == {"expon": [0.0]}
    assert tutorial.make_config(tutorial.CASES["baseline"]) == tutorial.CONFIG

    runner = tutorial.run_simulation(42, tutorial.CASES["baseline"])
    market = runner.simulator.name2market["Market"]
    assert set(market.get_fundamental_prices(times=range(600))) == {300.0}
    stats: Dict[str, float] = tutorial.deviation_stats(runner)
    # the first line of the table of seed 42 on the page
    assert stats["crosses"] == 77
    assert stats["volume"] == 207
    assert round(stats["std"], 2) == 5.73
    assert round(stats["max"], 2) == 14.99
    assert sum(market.get_executed_volumes(times=range(100))) == 0


def test_ci2002_tables(capsys: pytest.CaptureFixture[str]) -> None:
    tutorial = load_tutorial("tutorial_ci2002")
    results = tutorial.compare([42, 43])
    assert list(results) == ["baseline", "more fundamental", "more chart"]
    assert all(len(rows) == 2 for rows in results.values())
    means: Dict[str, Dict[str, float]] = tutorial.mean_rows(results)
    volumes: List[float] = [row["volume"] for row in results["baseline"]]
    assert means["baseline"]["volume"] == sum(volumes) / 2
    tutorial.print_table("title", means)
    lines: List[str] = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["title", text_blocks("ci2002.rst")[0][1]]
    assert [line.split()[-4:] for line in lines[2:]] == [
        [f"{row[key]:.2f}" for key in ("std", "max")]
        + [f"{row[key]:.1f}" for key in ("crosses", "volume")]
        for row in means.values()
    ]


def test_ci2002_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_tutorial("tutorial_ci2002")
    assert os.path.getsize(tmp_path / "ci2002_weights.png") > 0
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the two tables
    blocks: List[List[str]] = text_blocks("ci2002.rst")
    assert len(blocks) == 1
    assert_shown(blocks[0], lines)
    rows: Dict[str, List[float]] = {
        line[:16].strip(): [float(value) for value in line[16:].split()]
        for line in lines[7:10]
    }
    # the page says: more fundamental agents keep the price closer to the
    # fundamental price, and more chart agents make longer trends (fewer crosses)
    assert rows["more fundamental"][0] < rows["baseline"][0]
    assert rows["more fundamental"][1] < rows["baseline"][1]
    assert rows["more chart"][2] < rows["baseline"][2]
    assert rows["more chart"][0] < rows["baseline"][0]
    assert rows["more fundamental"][3] < rows["baseline"][3]
    assert rows["more chart"][3] < rows["baseline"][3]
