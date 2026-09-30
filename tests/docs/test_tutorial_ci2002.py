import copy
import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import matplotlib
import pytest

from pams.agents import FCNAgent
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


def test_ci2002_uses_sample() -> None:
    # the page says that the config is the one of samples/CI2002
    tutorial = load_tutorial("tutorial_ci2002")
    assert tutorial.CONFIG == load_sample_config("CI2002")


def test_ci2002_mean_reversion_time() -> None:
    # the page says: each agent gets a mean reversion time from 50 to 99 steps, and
    # without the key it is the agent's time window, from 100 to 199 steps
    tutorial = load_tutorial("tutorial_ci2002")
    config: Dict[str, Any] = copy.deepcopy(tutorial.CONFIG)
    assert config["FCNAgents"]["meanReversionTime"] == {"uniform": [50, 100]}
    assert config["FCNAgents"]["timeWindowSize"] == [100, 200]
    runner = SequentialRunner(settings=config, prng=random.Random(42))
    runner._setup()
    agents: List[FCNAgent] = [
        agent for agent in runner.simulator.agents if isinstance(agent, FCNAgent)
    ]
    assert len(agents) == 100
    assert min(agent.mean_reversion_time for agent in agents) >= 50
    assert max(agent.mean_reversion_time for agent in agents) <= 99
    del config["FCNAgents"]["meanReversionTime"]
    runner = SequentialRunner(settings=config, prng=random.Random(42))
    runner._setup()
    agents = [agent for agent in runner.simulator.agents if isinstance(agent, FCNAgent)]
    assert all(agent.mean_reversion_time == agent.time_window_size for agent in agents)
    assert min(agent.time_window_size for agent in agents) >= 100
    assert max(agent.time_window_size for agent in agents) <= 199
    assert_in_page(
        "ci2002.rst",
        "Each agent gets a value from 50 to 99 steps. Without it, :math:`\\tau_r` is the"
        " agent's ``timeWindowSize`` (100 to 199 steps).",
        "The configuration on the Plham tutorial page has no ``meanReversionTime``, so"
        " its agents use their ``timeWindowSize`` (100 to 199 steps) as"
        " :math:`\\tau_r`. The PAMS sample, like the sample of the current Plham, sets"
        " :math:`\\tau_r` to 50 to 99 steps.",
    )


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
    assert_same_image("ci2002_weights.png", tmp_path / "ci2002_weights.png")
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
    # with seed 42 alone, more chart has a smaller std than more fundamental, but
    # not on average
    seed_42: Dict[str, List[float]] = {
        line[:16].strip(): [float(value) for value in line[16:].split()]
        for line in lines[2:5]
    }
    assert seed_42["more chart"][0] < seed_42["more fundamental"][0]
    assert rows["more chart"][0] > rows["more fundamental"][0]
    base, fundamental, chart = (
        rows[name] for name in load_tutorial("tutorial_ci2002").CASES
    )
    assert_in_page(
        "ci2002.rst",
        f"``std`` falls from {base[0]:.2f} to {fundamental[0]:.2f} and ``max`` from"
        f" {base[1]:.2f} to {fundamental[1]:.2f}.",
        f"``crosses`` falls from {base[2]:.1f} to {chart[2]:.1f}:",
        f"``std`` does not grow: it falls from {base[0]:.2f} to {chart[0]:.2f}.",
        "``main()`` runs the three cases with the ten seeds 42 to 51.",
    )
