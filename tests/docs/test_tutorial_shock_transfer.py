import functools
import statistics
from pathlib import Path
from typing import Dict
from typing import List

import matplotlib
import pytest

from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import OrderLog
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


class TradeRecorder(Logger):
    """Keep the order logs and the execution logs of a run."""

    def __init__(self) -> None:
        """Initialize the lists of the logs."""
        super().__init__()
        self.orders: List[OrderLog] = []
        self.executions: List[ExecutionLog] = []

    def process_order_log(self, log: OrderLog) -> None:
        self.orders.append(log)

    def process_execution_log(self, log: ExecutionLog) -> None:
        self.executions.append(log)


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


def test_shock_transfer_seed_42(monkeypatch: pytest.MonkeyPatch) -> None:
    tutorial = load_tutorial("tutorial_shock_transfer")
    recorder = TradeRecorder()
    monkeypatch.setattr(
        tutorial,
        "SequentialRunner",
        functools.partial(SequentialRunner, logger=recorder),
    )
    runner = tutorial.run_simulation(42)
    measures: Dict[str, float] = tutorial.measure(runner)
    # the page says: the arbitrage agents sell 26 shares of spot 2 and buy 18 shares
    # of the index, the index market price is below the index value on average, and
    # spot 2 stays below 300
    assert measures["arb spot 2"] == -26
    assert measures["arb index"] == 18
    assert measures["premium"] < 0
    assert measures["spot 2"] < 0

    simulator = runner.simulator
    spot1 = simulator.name2market["SpotMarket-1"]
    spot2 = simulator.name2market["SpotMarket-2"]
    index = simulator.name2market["IndexMarket-I"]
    arbitrage: List[int] = [
        agent.agent_id for agent in simulator.agents_group_name2agent["ArbitrageAgents"]
    ]

    def ordered(market_id: int, is_buy: bool) -> int:
        return sum(
            log.volume
            for log in recorder.orders
            if log.agent_id in arbitrage
            and log.market_id == market_id
            and log.is_buy == is_buy
        )

    def executed(market_id: int, is_buy: bool) -> int:
        return sum(
            log.volume
            for log in recorder.executions
            if log.market_id == market_id
            and (log.buy_agent_id if is_buy else log.sell_agent_id) in arbitrage
        )

    # most of the orders of the arbitrage agents are not executed
    for market in [spot1, spot2, index]:
        for is_buy in [True, False]:
            assert (
                executed(market.market_id, is_buy)
                < ordered(market.market_id, is_buy) / 2
            )
    # they send an order to buy 2 shares of the index with each order to sell one
    # share of each stock, and the other way round, more often in the first way
    assert ordered(index.market_id, True) == 2 * ordered(spot1.market_id, False)
    assert ordered(index.market_id, False) == 2 * ordered(spot1.market_id, True)
    assert ordered(spot1.market_id, False) > ordered(spot1.market_id, True)
    spot1_net: int = executed(spot1.market_id, True) - executed(spot1.market_id, False)
    assert spot1_net == sum(
        agent.get_asset_volume(market_id=spot1.market_id) - 50
        for agent in simulator.agents_group_name2agent["ArbitrageAgents"]
    )

    spot1_prices: List[float] = spot1.get_market_prices()
    spot2_prices: List[float] = spot2.get_market_prices(times=range(300, 500))
    index_prices: List[float] = index.get_market_prices()
    lowest: float = min(spot2_prices)
    lowest_index: float = min(index_prices)
    assert_in_page(
        "shock_transfer.rst",
        f"Between steps 300 and 499, spot 2 is about"
        f" {statistics.mean(spot2_prices):.0f} on average and falls to {lowest:.2f} at"
        f" step {300 + spot2_prices.index(lowest)}, although its fundamental price stays"
        " at 300.",
        "In this run, spot 1 stays above its new fundamental price of 270 until step"
        f" {next(t for t, price in enumerate(spot1_prices) if price < 270)}, while the"
        " index falls below 285 at step"
        f" {next(t for t, price in enumerate(index_prices) if price < 285)} and down to"
        f" {lowest_index:.2f} at step {index_prices.index(lowest_index)}.",
        f"In total, they sell {-measures['arb spot 2']:.0f} shares of spot 2 and buy"
        f" {measures['arb index']:.0f} shares of the index.",
        f"They also end the run with {spot1_net} more shares of spot 1: their sell"
        f" orders for spot 1 add up to {ordered(spot1.market_id, False)} shares and"
        f" their buy orders to only {ordered(spot1.market_id, True)}, but more of the"
        f" buy orders are executed ({executed(spot1.market_id, True)} shares against"
        f" {executed(spot1.market_id, False)}).",
        "while in the run of seed 42 it is about"
        f" {300 - statistics.mean(spot2_prices):.0f} below 300 between steps 300 and"
        " 499 (see the plot).",
    )


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


def test_shock_transfer_other_seeds() -> None:
    # the page compares the cases with the seeds 52 to 61
    tutorial = load_tutorial("tutorial_shock_transfer")
    results = tutorial.compare(list(range(52, 62)))
    same, spot, index = (results[name] for name in tutorial.CASES)
    assert spot["spot 2"] < 0
    assert spot["spot 2"] > index["spot 2"]
    assert same["spot 2"] < index["spot 2"]
    assert spot["premium"] > 0 and spot["arb spot 2"] > 0
    assert index["premium"] < 0 and index["arb spot 2"] < 0
    assert_in_page(
        "shock_transfer.rst",
        f"With the seeds 52 to 61, spot 2 is {-spot['spot 2']:.2f} below 300 on"
        " average when spot 1 reacts faster, and above 300 in only"
        f" {spot['spot 2 up']:.0f} of the 10 seeds, but it is still higher than when"
        " the index reacts faster.",
    )


def test_shock_transfer_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    runners: List[SequentialRunner] = record_runners(monkeypatch)
    run_tutorial("tutorial_shock_transfer")
    assert_same_image(
        "shock_transfer_prices.png", tmp_path / "shock_transfer_prices.png"
    )
    # the run of seed 42, then the three cases with ten seeds and 100 steps each
    assert len(runners) == 31
    assert runners[0].simulator.sessions[1].iteration_steps == 500
    assert {runner.simulator.sessions[1].iteration_steps for runner in runners[1:]} == {
        100
    }
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
    assert all(abs(value) < 1 for value in spot2.values())
    assert_in_page(
        "shock_transfer.rst",
        "the arbitrage agents buy spot 2, and spot 2 is above 300 in 7 of the 10 seeds.",
        "In the table, spot 2 moves by less than one price unit on average,",
        "``compare`` runs each case with the ten seeds 42 to 51",
        "short, the second session has only 100 steps:",
    )
