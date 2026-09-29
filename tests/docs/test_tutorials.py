import copy
import csv
import importlib.util
import json
import os
import re
import runpy
from pathlib import Path
from types import ModuleType
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib
import pytest

from pams.agents import FCNAgent
from pams.runners import SequentialRunner

# the first tutorial imports matplotlib.pyplot: never open a window in the tests
matplotlib.use("Agg")

# the FCN agents place orders off the tick size, as the first tutorial explains
pytestmark = pytest.mark.filterwarnings(
    "ignore:order price does not accord to the tick size:UserWarning"
)

USER_GUIDE_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "docs",
    "source",
    "user_guide",
)
TUTORIAL_DIR: str = os.path.join(USER_GUIDE_DIR, "tutorials")
CODE_DIR: str = os.path.join(TUTORIAL_DIR, "code")

# Runner.main() prints the time taken, which changes from run to run
TIME_LABELS: Tuple[str, str] = ("# INITIALIZATION TIME ", "# EXECUTION TIME ")


def load_tutorial(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(CODE_DIR, f"{name}.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_tutorial(name: str) -> None:
    """Run a tutorial script as ``python <name>.py`` does."""
    runpy.run_path(os.path.join(CODE_DIR, f"{name}.py"), run_name="__main__")


def read_page(page: str) -> List[str]:
    """Read the lines of a tutorial page."""
    with open(os.path.join(TUTORIAL_DIR, page), encoding="utf-8") as fp:
        return fp.read().splitlines()


def text_blocks(page: str) -> List[List[str]]:
    """Read the outputs shown in the text code blocks of a tutorial page."""
    lines: List[str] = read_page(page)
    blocks: List[List[str]] = []
    for i, line in enumerate(lines):
        if line == ".. code-block:: text":
            assert lines[i + 1] == ""
            block: List[str] = []
            for block_line in lines[i + 2 :]:
                if not block_line.startswith("   "):
                    break
                block.append(block_line[3:])
            blocks.append(block)
    return blocks


def mask_time(line: str) -> str:
    """Keep only the label of a time line."""
    for label in TIME_LABELS:
        if line.startswith(label):
            return label
    return line


def assert_shown(shown: List[str], output: List[str]) -> None:
    """Check that the output shown in a page is the actual output.

    A line "..." in the page stands for any number of lines, and only the labels of
    the time lines are compared.
    """
    expected: List[str] = [mask_time(line) for line in shown]
    actual: List[str] = [mask_time(line) for line in output]
    if "..." not in expected:
        assert actual == expected
        return
    segments: List[List[str]] = [[]]
    for line in expected:
        if line == "...":
            segments.append([])
        else:
            segments[-1].append(line)
    first, *middle, last = segments
    assert actual[: len(first)] == first
    assert actual[len(actual) - len(last) :] == last
    position: int = len(first)
    for segment in middle:
        # find the segment after the previous one
        for start in range(position, len(actual) - len(last) - len(segment) + 1):
            if actual[start : start + len(segment)] == segment:
                position = start + len(segment)
                break
        else:
            pytest.fail(f"not in the output: {segment}")
    assert position <= len(actual) - len(last)


def test_first_simulation_uses_minimal_example() -> None:
    # the tutorial says that its config is the minimal config of the Quick start
    tutorial = load_tutorial("tutorial_first_simulation")
    path = os.path.join(USER_GUIDE_DIR, "config_examples", "minimal.json")
    with open(path, encoding="utf-8") as fp:
        assert tutorial.CONFIG == json.load(fp)


def test_first_simulation_print(capsys: pytest.CaptureFixture[str]) -> None:
    tutorial = load_tutorial("tutorial_first_simulation")
    with pytest.warns(UserWarning) as record:
        tutorial.print_prices(seed=42)
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the output of seed 42 and the warning of the tick size
    blocks: List[List[str]] = text_blocks("first_simulation.rst")
    assert_shown(blocks[0], lines)
    assert blocks[1] == [f"UserWarning: {record[0].message}"]
    steps: List[List[str]] = [line.split() for line in lines[:-2]]
    assert len(steps) == 600
    assert [int(step[0]) for step in steps] == [0] * 100 + [1] * 500
    assert [int(step[1]) for step in steps] == list(range(600))
    assert {step[3] for step in steps} == {"Market"}
    # the price does not move while orders are not executed
    assert {float(step[4]) for step in steps[:100]} == {300.0}
    # no fundamentalVolatility: the fundamental price does not move
    assert {float(step[5]) for step in steps} == {300.0}

    # the same seed gives the same prices as the saver
    _, saver = tutorial.run_simulation(seed=42)
    assert [float(step[4]) for step in steps] == [
        log["market_price"] for log in saver.market_step_logs
    ]


def test_first_simulation_results(tmp_path: Path) -> None:
    tutorial = load_tutorial("tutorial_first_simulation")
    runner, saver = tutorial.run_simulation(seed=42)
    summary: Dict[str, float] = tutorial.summarize(runner)
    assert summary["min_price"] <= summary["last_price"] <= summary["max_price"]
    assert summary["total_volume"] > 0
    assert summary["total_shares"] == 100 * 50
    assert tutorial.summarize(tutorial.run_simulation(seed=42)[0]) == summary
    assert len(saver.market_step_logs) == 600
    assert saver.market_step_logs[100]["session_id"] == 1
    assert saver.market_step_logs[100]["market_time"] == 100
    # the page shows this log of seed 42
    page: List[str] = read_page("first_simulation.rst")
    shown: str = page[page.index("   >>> saver.market_step_logs[100]") + 1]
    assert str(saver.market_step_logs[100]) == shown.strip()
    # after the run, the market is one step after the last one
    assert runner.simulator.name2market["Market"].get_time() == 600

    path = str(tmp_path / "prices.png")
    tutorial.plot_prices(saver, path=path)
    assert os.path.getsize(path) > 0


def test_first_simulation_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_tutorial("tutorial_first_simulation")
    assert os.path.getsize(tmp_path / "prices.png") > 0
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the summary of seed 42
    assert lines[-1:] == text_blocks("first_simulation.rst")[2]


def test_custom_agent() -> None:
    tutorial = load_tutorial("tutorial_custom_agent")
    runner = tutorial.run_simulation(seed=42)
    simulator = runner.simulator
    market = simulator.name2market["Market"]
    moving_average_agents = simulator.agents_group_name2agent["MovingAverageAgents"]
    assert len(moving_average_agents) == 10
    for agent in moving_average_agents:
        assert isinstance(agent, tutorial.MovingAverageAgent)
        assert 10 <= agent.window_size < 30
        assert agent.order_volume == 1
        # the callbacks saw every execution of the agent
        assert agent.get_asset_volume(market_id=market.market_id) == 50 + sum(
            volume for _, _, volume in agent.trades
        )
        assert agent.get_cash_amount() == pytest.approx(
            10000 - sum(price * volume for _, price, volume in agent.trades)
        )
        # each cancel is for an order placed before
        assert agent.n_canceled <= max(agent.n_submitted - 1, 0)
    assert sum(agent.n_submitted for agent in moving_average_agents) > 0
    assert sum(len(agent.trades) for agent in moving_average_agents) > 0
    assert sum(agent.n_canceled for agent in moving_average_agents) > 0

    contrarian_agents = simulator.agents_group_name2agent["ContrarianAgents"]
    assert len(contrarian_agents) == 20
    for agent in contrarian_agents:
        assert isinstance(agent, tutorial.ContrarianFCNAgent)
        assert agent.is_chart_following is False
        assert agent.chart_weight > 0.0
    fcn_agents = simulator.agents_group_name2agent["FCNAgents"]
    assert len(fcn_agents) == 100
    for agent in fcn_agents:
        assert isinstance(agent, FCNAgent)
        assert not isinstance(agent, tutorial.ContrarianFCNAgent)
        assert agent.is_chart_following is True
        assert agent.chart_weight == 0.0
    assert (
        sum(
            agent.get_asset_volume(market_id=market.market_id)
            for agent in simulator.agents
        )
        == 130 * 50
    )


def test_custom_agent_inaccessible_market(monkeypatch: pytest.MonkeyPatch) -> None:
    tutorial = load_tutorial("tutorial_custom_agent")
    config = copy.deepcopy(tutorial.CONFIG)
    # the FCN agents also trade in a second market, which the others cannot access
    config["simulation"]["markets"].append("OtherMarket")
    config["OtherMarket"] = copy.deepcopy(config["Market"])
    config["FCNAgents"]["markets"].append("OtherMarket")
    monkeypatch.setattr(tutorial, "CONFIG", config)
    runner = tutorial.run_simulation(seed=42)
    simulator = runner.simulator
    market = simulator.name2market["Market"]
    other_market = simulator.name2market["OtherMarket"]
    assert sum(other_market.get_executed_volumes()) > 0
    moving_average_agents = simulator.agents_group_name2agent["MovingAverageAgents"]
    for agent in moving_average_agents:
        assert isinstance(agent, tutorial.MovingAverageAgent)
        assert not agent.is_market_accessible(market_id=other_market.market_id)
        # the agent places orders only in the market it can access
        assert set(agent.last_orders) <= {market.market_id}
    assert sum(len(agent.last_orders) for agent in moving_average_agents) > 0


def test_custom_agent_main(capsys: pytest.CaptureFixture[str]) -> None:
    run_tutorial("tutorial_custom_agent")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the report of seed 42 after the two time lines
    assert_shown(list(TIME_LABELS) + text_blocks("custom_agent.rst")[0], lines)


def test_custom_event(capsys: pytest.CaptureFixture[str]) -> None:
    tutorial = load_tutorial("tutorial_custom_event")
    runner = tutorial.run_simulation(seed=42)
    news = runner.simulator.name2event["News"]
    assert isinstance(news, tutorial.NewsShock)
    assert news.news_times == list(range(150, 600, 50))
    market = runner.simulator.name2market["Market"]
    fundamental_prices: List[float] = market.get_fundamental_prices()
    for time in range(1, 600):
        # the fundamental price moves only with the news (no fundamentalVolatility)
        changed: bool = fundamental_prices[time] != fundamental_prices[time - 1]
        assert changed == (time in news.news_times)
    out: str = capsys.readouterr().out
    lines: List[str] = out.splitlines()
    # the session hook is called at the end of main only, before main() prints times
    assert re.fullmatch(r"News: 9 news in main, fundamental price \d+\.\d\d", lines[0])
    assert lines[1].startswith("# INITIALIZATION TIME ")


def test_custom_event_main(capsys: pytest.CaptureFixture[str]) -> None:
    run_tutorial("tutorial_custom_event")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the output of seed 42
    assert_shown(text_blocks("custom_event.rst")[0], lines)


@pytest.mark.parametrize("key", ["target", "interval", "jumpScale"])
def test_custom_event_required_keys(key: str) -> None:
    tutorial = load_tutorial("tutorial_custom_event")
    config = copy.deepcopy(tutorial.CONFIG)
    del config["News"][key]
    runner = SequentialRunner(settings=config)
    runner.class_register(cls=tutorial.NewsShock)
    with pytest.raises(ValueError, match=f"^{key} is required for NewsShock$"):
        runner.main()


def test_custom_logger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tutorial = load_tutorial("tutorial_custom_logger")
    path = str(tmp_path / "trades.csv")
    runner = tutorial.run_simulation(seed=42, path=path)
    market = runner.simulator.name2market["Market"]
    total_volume: int = sum(market.get_executed_volumes())
    assert total_volume > 0

    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the output of seed 42 and the head of the CSV file
    blocks: List[List[str]] = text_blocks("custom_logger.rst")
    assert_shown(blocks[0], lines)
    with open(path, encoding="utf-8") as fp:
        assert fp.read().splitlines()[:4] == blocks[1]
    # one FCN agent places one order in each step
    assert lines[0] == "warmup: 100 orders, 0 shares traded"
    assert lines[1] == f"main: 500 orders, {total_volume} shares traded"

    with open(path, newline="", encoding="utf-8") as fp:
        rows: List[Dict[str, str]] = list(csv.DictReader(fp))
    assert sum(int(row["volume"]) for row in rows) == total_volume
    assert all(int(row["time"]) >= 100 for row in rows)
    assert int(rows[0]["time"]) == 100
    assert {row["market"] for row in rows} == {"Market"}
    assert all(row["buyer"].startswith("FCNAgents-") for row in rows)
    assert all(row["seller"].startswith("FCNAgents-") for row in rows)
    # executions are logged in the order they happened
    assert [int(row["time"]) for row in rows] == sorted(
        int(row["time"]) for row in rows
    )


def test_custom_logger_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_tutorial("tutorial_custom_logger")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the summaries and the time lines, then the head of the CSV file
    blocks: List[List[str]] = text_blocks("custom_logger.rst")
    assert_shown(blocks[0] + blocks[1], lines)
