import random
import statistics
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import matplotlib
import pytest

from pams import Market
from pams.agents import MarketShareFCNAgent
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


def last_trade(runner: SequentialRunner, name: str) -> int:
    """Find the last step in which a market trades."""
    volumes: List[int] = runner.simulator.name2market[name].get_executed_volumes()
    return max(time for time, volume in enumerate(volumes) if volume > 0)


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


def test_market_share_volume_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # the page says: the agents sum the executed volume of the steps t - tau to t,
    # the current step included, and add 1e-10 to each weight
    tutorial = load_tutorial("tutorial_market_share")
    runner: SequentialRunner = tutorial.run_simulation(
        tutorial.CONFIG_TICK, seed=42, main_steps=30
    )
    agent = next(
        agent
        for agent in runner.simulator.agents
        if isinstance(agent, MarketShareFCNAgent)
    )
    markets: List[Market] = [
        runner.simulator.name2market[name] for name in ["Market-A", "Market-B"]
    ]
    # go back to the last step in which Market-A trades
    volumes: List[int] = markets[0].get_executed_volumes()
    t: int = max(time for time, volume in enumerate(volumes) if volume > 0)
    for market in markets:
        monkeypatch.setattr(market, "get_time", lambda: t)
    tau: int = min(agent.time_window_size, t)
    window = range(t - tau, t + 1)
    sums: List[int] = [
        sum(market.get_executed_volumes(times=window)) for market in markets
    ]
    assert agent.get_sum_trade_volume(market=markets[0]) == sums[0]
    assert sums[0] > sum(markets[0].get_executed_volumes(times=range(t - tau, t)))
    chosen: List[List[float]] = []

    def choices(population: List[Market], weights: List[float]) -> List[Market]:
        chosen.append(weights)
        return [population[0]]

    monkeypatch.setattr(agent.get_prng(), "choices", choices)
    agent.submit_orders(markets=markets)
    assert chosen == [[sums[0] + 1e-10, sums[1] + 1e-10]]
    assert_in_page(
        "market_share.rst",
        ":class:`~pams.agents.MarketShareFCNAgent` sums the executed volume of the steps"
        " :math:`t - \\tau` to :math:`t`, the current step included, and adds"
        " :math:`10^{-10}` to each weight.",
    )


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
    runners: List[SequentialRunner] = record_runners(monkeypatch)
    run_tutorial("tutorial_market_share")
    lines: List[str] = capsys.readouterr().out.splitlines()
    # the page shows the whole output, in three blocks
    blocks: List[List[str]] = text_blocks("market_share.rst")
    assert len(blocks) == 3
    assert_shown([line for block in blocks for line in block], lines)
    for name in ["market_share_tick.png", "market_share_mm.png"]:
        assert_same_image(name, tmp_path / name)
    # two runs of seed 42, then 3 tick sizes and 4 spreads with ten seeds each
    assert len(runners) == 2 + 3 * 10 + 4 * 10
    tutorial = load_tutorial("tutorial_market_share")

    # the two runs of seed 42
    tick_10: SequentialRunner = runners[0]
    tick_2: SequentialRunner = runners[1]
    shares_10: List[float] = tutorial.share_of_b(tick_10)
    shares_2: List[float] = tutorial.share_of_b(tick_2)
    prices_a: List[float] = tick_10.simulator.name2market[
        "Market-A"
    ].get_market_prices()
    first_310: int = prices_a.index(310.0)
    assert set(prices_a[:first_310]) == {300.0}
    assert set(prices_a[first_310:]) == {310.0}
    assert round(statistics.mean(shares_10[:4]), 2) == 0.05
    assert shares_10[8] == 1.0 and shares_2[5] == 0.0 and shares_2[4] > 0.0
    assert_in_page(
        "market_share.rst",
        f"The price of Market-A is 300, then 310 from step {first_310}, and then stops"
        " changing,",
        "With a tick size of 10, ``Market-B`` keeps about 5% of the volume until step"
        " 499, then takes all of it: after step"
        f" {last_trade(tick_10, 'Market-A')}, ``Market-A`` has no trade, and its price"
        f" stays at {prices_a[-1]:.0f}, one of the only two prices at which it traded"
        " (300 and 310). With a tick size of 2 and the same seed, the opposite happens:"
        f" ``Market-B`` has no trade after step {last_trade(tick_2, 'Market-B')}.",
    )

    # the table of the tick sizes, with the seeds 42 to 51 in order
    ticks: Dict[str, List[str]] = {
        line.split()[0]: line.split()[1:] for line in blocks[1][1:]
    }
    assert ticks["2"][0] == "0/10"
    assert_in_page(
        "market_share.rst",
        "the more often ``Market-B`` takes the volume: in"
        f" {ticks['10'][0].split('/')[0]} of 10 runs with a tick size of 10, in"
        f" {ticks['5'][0].split('/')[0]} with 5 and in none with 2.",
        f"with seed {tutorial.SEEDS[9]}, ``Market-B`` has {ticks['5'][11]} of the volume"
        f" with a tick size of 5 but only {ticks['10'][11]} with 10,",
        "PAMS finds the same direction, but ``Market-B`` wins only in"
        f" {ticks['5'][0].split('/')[0]} of 10 runs with a tick size of 5.",
    )

    # the table of the spreads
    spreads: Dict[str, List[str]] = {
        line.split()[0]: line.split()[1:] for line in blocks[2][1:]
    }
    none: List[float] = [float(share) for share in spreads.pop("none")[2:]]
    assert blocks[2][1].split()[1:3] == ["0/10", "0.10"]
    means: List[float] = [float(row[1]) for row in spreads.values()]
    assert means == sorted(means)
    for row in spreads.values():
        by_seed: List[float] = [float(share) for share in row[2:]]
        # seed 42 is the exception
        assert by_seed[0] == 0.0 and none[0] == 0.0
        assert all(
            share > max(0.1, before) for share, before in zip(by_seed[1:], none[1:])
        )
    assert_in_page(
        "market_share.rst",
        "Without the market maker it stays near 0.1. With the market maker it rises in"
        f" every case, to about {means[0]:.2f} with a spread of 0.02, {means[1]:.2f}"
        f" with 0.01 and {means[2]:.2f} with 0.0001.",
        "Without the market maker, ``Market-B`` keeps about its initial 10% on average"
        " and wins in no run.",
        "Seed 42 is the exception: with every spread, ``Market-B`` has no trade at all"
        " in the second session.",
    )
