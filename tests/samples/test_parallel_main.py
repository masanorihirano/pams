import argparse
import copy
import csv
import json
import math
import os.path
import random
import runpy
import statistics
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from unittest import mock

import pytest

from pams.market import Market
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners import SequentialRunner
from samples.parallel_main import benchmark
from samples.parallel_main.main import main
from samples.parallel_main.make_config import main as make_config_main
from samples.parallel_main.make_config import make_config
from samples.parallel_main.workload_fcn_agent import DEFAULT_ORDER_RATE
from samples.parallel_main.workload_fcn_agent import WorkloadFCNAgent
from samples.parallel_main.workload_fcn_agent import black_scholes_monte_carlo

ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAMPLE_DIR: str = os.path.join(ROOT_DIR, "samples", "parallel_main")
RUNNER_NAMES: List[str] = ["sequential", "multi_thread", "multi_process"]
EXTRA_KEYS: List[str] = ["orderRate", "bsWorkload", "bsNumSamples", "bsNumSteps"]


def small_config(**kwargs: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "n_spots": 2,
        "steps": 15,
        "agents_per_market": 15,
        "arbitrage_agents": 5,
    }
    params.update(kwargs)
    return make_config(**params)


def make_runner(
    settings: Dict[str, Any],
    runner_class: Type[SequentialRunner] = SequentialRunner,
    seed: int = 42,
) -> SequentialRunner:
    runner = runner_class(
        settings=copy.deepcopy(settings), prng=random.Random(seed), logger=None
    )
    runner.class_register(cls=WorkloadFCNAgent)
    return runner


def run(
    settings: Dict[str, Any],
    runner_class: Type[SequentialRunner] = SequentialRunner,
    seed: int = 42,
) -> List[List[float]]:
    runner = make_runner(settings=settings, runner_class=runner_class, seed=seed)
    runner.main()
    results: List[List[float]] = []
    for market in runner.simulator.markets:
        times = range(market.get_time() + 1)
        results.append(market.get_market_prices(times=times))
        results.append(market.get_fundamental_prices(times=times))
        results.append([float(v) for v in market.get_executed_volumes(times=times)])
    for agent in runner.simulator.agents:
        volumes = [
            float(agent.asset_volumes[key]) for key in sorted(agent.asset_volumes)
        ]
        results.append([agent.cash_amount] + volumes)
    return results


def run_main(runner_name: str, capsys: pytest.CaptureFixture[str]) -> List[str]:
    argv = [
        "main.py",
        "--config",
        f"{SAMPLE_DIR}/config.json",
        "--seed",
        "1",
        "--runner",
        runner_name,
    ]
    with mock.patch("sys.argv", argv):
        main()
    out: str = capsys.readouterr().out
    return [line for line in out.splitlines() if not line.startswith("#")]


@pytest.mark.parametrize("runner_name", RUNNER_NAMES)
def test_parallel_main(runner_name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert len(run_main(runner_name=runner_name, capsys=capsys)) > 0


def test_main_output_is_the_same_for_all_runners(
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs = [run_main(runner_name=name, capsys=capsys) for name in RUNNER_NAMES]
    assert outputs[0] == outputs[1] == outputs[2]


class RecordingRunner(MultiProcessAgentParallelRunner):
    instances: List[MultiProcessAgentParallelRunner] = []

    def _setup(self) -> None:
        super()._setup()
        RecordingRunner.instances.append(self)


def test_main_num_parallel(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["main.py", "-c", f"{SAMPLE_DIR}/config.json", "-s", "1", "-n", "3"]
    RecordingRunner.instances.clear()
    with mock.patch.dict(
        "samples.parallel_main.main.RUNNERS", {"multi_process": RecordingRunner}
    ), mock.patch("sys.argv", argv):
        main()
    assert len(capsys.readouterr().out) > 0
    assert [runner.num_parallel for runner in RecordingRunner.instances] == [3]


@pytest.mark.parametrize(
    "runner_class, num_parallel",
    [
        (MultiThreadAgentParallelRunner, 2),
        (MultiProcessAgentParallelRunner, 2),
        (MultiProcessAgentParallelRunner, 3),
    ],
)
def test_same_results_as_sequential(
    runner_class: Type[SequentialRunner], num_parallel: int
) -> None:
    settings = small_config(num_parallel=num_parallel)
    assert run(settings, runner_class) == run(settings)


def test_workload_does_not_change_results() -> None:
    assert run(small_config(with_workload=True)) == run(
        small_config(with_workload=False)
    )


def test_order_rate_one_is_fcn_agent() -> None:
    settings = small_config(order_rate=1.0, with_workload=True)
    fcn_settings = copy.deepcopy(settings)
    fcn_settings["FCNAgent"]["class"] = "FCNAgent"
    for key in EXTRA_KEYS:
        del fcn_settings["FCNAgent"][key]
    assert run(settings) == run(fcn_settings)


def test_order_rate_changes_results() -> None:
    assert run(small_config(order_rate=0.1)) != run(small_config(order_rate=1.0))


@pytest.mark.parametrize("with_workload", [True, False])
def test_workload_is_computed(with_workload: bool) -> None:
    original = WorkloadFCNAgent.compute_workload
    prices: List[float] = []

    def spy(self: WorkloadFCNAgent, market: Market) -> float:
        price = original(self, market=market)
        prices.append(price)
        return price

    with mock.patch.object(WorkloadFCNAgent, "compute_workload", spy):
        run(small_config(with_workload=with_workload))
    n_agents: int = 2 * 15 + 15
    n_steps: int = 15
    if with_workload:
        assert len(prices) == n_agents * n_steps
        assert 20.0 < statistics.mean(prices) < 50.0
    else:
        assert len(prices) == 0


def test_compute_workload_is_deterministic() -> None:
    runner = make_runner(settings=small_config())
    runner._setup()
    agent = runner.simulator.normal_frequency_agents[0]
    assert isinstance(agent, WorkloadFCNAgent)
    market = runner.simulator.markets[0]
    state = agent.prng.getstate()
    price = agent.compute_workload(market=market)
    assert agent.compute_workload(market=market) == price
    assert agent.prng.getstate() == state
    later = mock.Mock(spec=Market, market_id=market.market_id)
    later.get_time.return_value = market.get_time() + 1
    assert agent.compute_workload(market=later) != price


def test_default_parameters() -> None:
    settings = small_config()
    for key in EXTRA_KEYS:
        del settings["FCNAgent"][key]
    runner = make_runner(settings=settings)
    runner._setup()
    agents = runner.simulator.normal_frequency_agents
    assert all(isinstance(agent, WorkloadFCNAgent) for agent in agents)
    for agent in agents:
        assert isinstance(agent, WorkloadFCNAgent)
        assert agent.order_rate == DEFAULT_ORDER_RATE
        assert not agent.bs_workload
        assert agent.bs_n_samples == 0
        assert agent.bs_n_steps == 0


def test_black_scholes_monte_carlo() -> None:
    price = black_scholes_monte_carlo(
        prng=random.Random(1), n_samples=20000, n_steps=50
    )
    assert price == pytest.approx(33.6044837628, abs=1.0)
    with pytest.raises(ValueError):
        black_scholes_monte_carlo(prng=random.Random(1), n_samples=0, n_steps=1)
    with pytest.raises(ValueError):
        black_scholes_monte_carlo(prng=random.Random(1), n_samples=1, n_steps=0)


@pytest.mark.parametrize(
    "updates, removed",
    [
        ({"bsWorkload": "true"}, []),
        ({"bsWorkload": True}, ["bsNumSamples"]),
        ({"bsNumSamples": 0}, []),
        ({"bsNumSamples": True}, []),
        ({"bsNumSamples": 1.5}, []),
        ({"bsWorkload": False, "bsNumSteps": -1}, []),
        ({"orderRate": 1.5}, []),
        ({"orderRate": -0.1}, []),
    ],
)
def test_setup_errors(updates: Dict[str, Any], removed: List[str]) -> None:
    settings = small_config()
    settings["FCNAgent"].update(updates)
    for key in removed:
        del settings["FCNAgent"][key]
    runner = make_runner(settings=settings)
    with pytest.raises(ValueError):
        runner._setup()


@pytest.mark.parametrize("n_spots", [2, 9, 99])
def test_checked_in_configs(n_spots: int) -> None:
    path = os.path.join(SAMPLE_DIR, f"config-{n_spots:03d}.json")
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == make_config(n_spots=n_spots)


def test_fast_config() -> None:
    with open(os.path.join(SAMPLE_DIR, "config.json"), encoding="utf-8") as f:
        assert json.load(f) == make_config(
            n_spots=2,
            steps=20,
            agents_per_market=20,
            arbitrage_agents=10,
            num_parallel=2,
        )


def test_make_config_cli(capsys: pytest.CaptureFixture[str], tmp_path: Any) -> None:
    make_config_main(["9"])
    assert json.loads(capsys.readouterr().out) == make_config(n_spots=9)

    options = [
        "3",
        "--steps",
        "7",
        "--agents-per-market",
        "6",
        "--arbitrage-agents",
        "2",
        "--order-rate",
        "0.5",
        "--no-workload",
        "--bs-num-samples",
        "3",
        "--bs-num-steps",
        "4",
        "--no-shock",
        "--num-parallel",
        "4",
    ]
    make_config_main(options)
    expected = make_config(
        n_spots=3,
        steps=7,
        agents_per_market=6,
        arbitrage_agents=2,
        order_rate=0.5,
        with_workload=False,
        bs_num_samples=3,
        bs_num_steps=4,
        with_shock=False,
        num_parallel=4,
    )
    config = json.loads(capsys.readouterr().out)
    assert config == expected
    assert config["simulation"]["numParallel"] == 4
    assert config["simulation"]["sessions"][0]["iterationSteps"] == 7
    assert config["FundamentalPriceShock"]["enabled"] is False
    assert config["FCNAgent"]["bsWorkload"] is False
    assert config["FCNAgent"]["bsNumSamples"] == 3
    assert config["FCNAgent"]["bsNumSteps"] == 4

    output = os.path.join(str(tmp_path), "config.json")
    make_config_main(["9", "-o", output])
    assert capsys.readouterr().out == ""
    with open(output, "rb") as fb:
        data: bytes = fb.read()
    assert b"\r" not in data
    assert data.endswith(b"}\n")
    with open(os.path.join(SAMPLE_DIR, "config-009.json"), "rb") as fb:
        assert data == fb.read().replace(b"\r\n", b"\n")

    with pytest.raises(ValueError):
        make_config(n_spots=0)


@pytest.mark.parametrize(
    "script, args",
    [
        ("make_config.py", ["2", "--steps", "3"]),
        (
            "main.py",
            ["--config", f"{SAMPLE_DIR}/config.json", "--runner", "sequential"],
        ),
        (
            "benchmark.py",
            [
                "--spots",
                "1",
                "--steps",
                "2",
                "--agents-per-market",
                "2",
                "--arbitrage-agents",
                "1",
                "--runners",
                "sequential",
                "--workloads",
                "off",
            ],
        ),
    ],
)
def test_run_as_script(
    script: str, args: List[str], capsys: pytest.CaptureFixture[str]
) -> None:
    path = os.path.join(SAMPLE_DIR, script)
    with mock.patch("sys.argv", [path] + args):
        runpy.run_path(path, run_name="__main__")
    assert capsys.readouterr().out != ""


def test_benchmark(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    output = os.path.join(str(tmp_path), "benchmark.csv")
    image = os.path.join(str(tmp_path), "benchmark.png")
    benchmark.main(
        [
            "--spots",
            "2",
            "--steps",
            "3",
            "--warmup",
            "1",
            "--agents-per-market",
            "5",
            "--arbitrage-agents",
            "2",
            "--num-parallel",
            "1",
            "2",
            "--runners",
            "sequential",
            "multi_thread",
            "multi_process",
            "--workloads",
            "off",
            "2x2",
            "--output",
            output,
            "--plot",
            image,
        ]
    )
    assert len(capsys.readouterr().out.splitlines()) == 10
    with open(output, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == benchmark.COLUMNS
        rows = list(reader)
    assert len(rows) == 2 * (1 + 2 + 2)
    assert [row["workload"] for row in rows] == ["off"] * 5 + ["2x2"] * 5
    assert [row["numParallel"] for row in rows] == ["0", "1", "2", "1", "2"] * 2
    assert len({row["digest"] for row in rows}) == 1
    for row in rows:
        for key in ["total", "first_step", "step", "collect", "handle"]:
            value = float(row[key])
            assert math.isfinite(value)
            assert value >= 0.0
    assert os.path.getsize(image) > 0

    benchmark.check_digests(rows=rows)
    rows[-1]["digest"] = "different"
    with pytest.raises(RuntimeError):
        benchmark.check_digests(rows=rows)


def test_benchmark_options() -> None:
    assert benchmark.parse_workload("off") is None
    assert benchmark.parse_workload("10x20") == (10, 20)
    for value in ["0x5", "5x0", "abc", "1x2x3", "-1x5"]:
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark.parse_workload(value)
    assert benchmark.format_workload(None) == "off"
    assert benchmark.format_workload((10, 20)) == "10x20"
    assert benchmark._format_major_tick(0.01, None) == "0.01"
    assert benchmark._format_minor_tick(0.002, None) == "0.002"
    assert benchmark._format_minor_tick(50.0, None) == "50"
    assert benchmark._format_minor_tick(0.003, None) == ""
    assert benchmark._format_minor_tick(0.0, None) == ""
    with pytest.raises(SystemExit):
        benchmark.main(["--steps", "1", "--warmup", "1"])
