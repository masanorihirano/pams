import copy
import json
import math
import os.path
import pathlib
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from unittest import mock

import pytest

from pams import Market
from pams import Order
from pams import Simulator
from pams.logs.market_step_loggers import MarketStepSaver
from pams.runners import MultiProcessAgentParallelRunner
from pams.runners import MultiThreadAgentParallelRunner
from pams.runners import SequentialRunner
from samples.black_scholes.black_scholes import BlackScholes
from samples.black_scholes.main import RUNNERS
from samples.black_scholes.main import main
from samples.black_scholes.workload_fcn_agent import BS_INITIAL_PRICE
from samples.black_scholes.workload_fcn_agent import BS_MATURITY_TIME
from samples.black_scholes.workload_fcn_agent import BS_RISK_FREE_RATE
from samples.black_scholes.workload_fcn_agent import BS_STRIKE_PRICE
from samples.black_scholes.workload_fcn_agent import BS_VOLATILITY
from samples.black_scholes.workload_fcn_agent import DEFAULT_ORDER_RATE
from samples.black_scholes.workload_fcn_agent import WorkloadFCNAgent

ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAMPLE_DIR: str = os.path.join(ROOT_DIR, "samples", "black_scholes")

# the theoretical price printed by BlackScholes.main of plhamJ
THEORETICAL_PRICE: float = 33.6044837628


def _create_black_scholes(
    seed: int, volatility: float = BS_VOLATILITY, strike_price: float = BS_STRIKE_PRICE
) -> BlackScholes:
    return BlackScholes(
        prng=random.Random(seed),
        initial_price=BS_INITIAL_PRICE,
        strike_price=strike_price,
        risk_free_rate=BS_RISK_FREE_RATE,
        volatility=volatility,
        maturity_time=BS_MATURITY_TIME,
    )


class TestBlackScholes:
    def test_theoretical_price(self) -> None:
        black_scholes = _create_black_scholes(seed=42)
        assert black_scholes.theoretical_price() == pytest.approx(
            THEORETICAL_PRICE, abs=1e-9
        )

    @pytest.mark.parametrize("seed", [1, 42])
    def test_compute(self, seed: int) -> None:
        black_scholes = _create_black_scholes(seed=seed)
        price = black_scholes.compute(n_samples=2000, n_steps=50)
        assert price == pytest.approx(THEORETICAL_PRICE, abs=3.0)
        # the same seed gives the same estimate
        assert (
            _create_black_scholes(seed=seed).compute(n_samples=2000, n_steps=50)
            == price
        )

    @pytest.mark.parametrize("n_samples", [1, 3])
    @pytest.mark.parametrize("n_steps", [1, 4])
    def test_compute_random_numbers(self, n_samples: int, n_steps: int) -> None:
        # one gaussian random number is drawn for each time step of each sample path
        black_scholes = _create_black_scholes(seed=42)
        black_scholes.compute(n_samples=n_samples, n_steps=n_steps)
        prng = random.Random(42)
        for _ in range(n_samples * n_steps):
            prng.gauss(mu=0.0, sigma=1.0)
        assert black_scholes.prng.getstate() == prng.getstate()

    @pytest.mark.parametrize("n_steps", [1, 10, 100])
    def test_compute_without_volatility(self, n_steps: int) -> None:
        # the price grows deterministically at the risk-free rate by the euler method
        dt = BS_MATURITY_TIME / n_steps
        final_price = BS_INITIAL_PRICE * (1 + BS_RISK_FREE_RATE * dt) ** n_steps
        discount = math.exp(-BS_RISK_FREE_RATE * BS_MATURITY_TIME)
        black_scholes = _create_black_scholes(seed=42, volatility=0.0)
        assert black_scholes.compute(n_samples=3, n_steps=n_steps) == pytest.approx(
            discount * (final_price - BS_STRIKE_PRICE)
        )
        black_scholes = _create_black_scholes(
            seed=42, volatility=0.0, strike_price=final_price + 1.0
        )
        assert black_scholes.compute(n_samples=3, n_steps=n_steps) == 0.0

    def test_compute_invalid(self) -> None:
        black_scholes = _create_black_scholes(seed=42)
        with pytest.raises(ValueError):
            black_scholes.compute(n_samples=0, n_steps=10)
        with pytest.raises(ValueError):
            black_scholes.compute(n_samples=10, n_steps=0)


class TestWorkloadFCNAgent:
    settings: Dict[str, Any] = {
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": 1.0,
        "chartWeight": 2.0,
        "noiseWeight": 3.0,
        "noiseScale": 0.001,
        "timeWindowSize": 100,
        "orderMargin": 0.1,
        "marginType": "fixed",
        "meanReversionTime": 200,
    }

    def _create_agent(self, seed: int, **kwargs: Any) -> WorkloadFCNAgent:
        sim = Simulator(prng=random.Random(seed + 1))
        agent = WorkloadFCNAgent(
            agent_id=1, prng=random.Random(seed), simulator=sim, name="test_agent"
        )
        settings = copy.deepcopy(self.settings)
        settings.update(kwargs)
        agent.setup(settings=settings, accessible_markets_ids=[0])
        return agent

    def _create_market(
        self, agent: WorkloadFCNAgent, market_id: int, time: int = 0
    ) -> Market:
        market = Market(
            market_id=market_id,
            prng=random.Random(market_id),
            simulator=agent.simulator,
            name=f"market{market_id}",
        )
        for _ in range(time + 1):
            market._update_time(next_fundamental_price=300.0)
        assert market.get_time() == time
        return market

    def test_setup(self) -> None:
        agent = self._create_agent(seed=42)
        assert agent.bs_workload is False
        assert agent.bs_load_shutdown == -1
        assert agent.bs_n_samples == 0
        assert agent.bs_n_steps == 0
        assert agent.order_rate == DEFAULT_ORDER_RATE == 0.1
        assert agent.bs_sum == 0.0
        assert agent.time_window_size == 100

        agent = self._create_agent(
            seed=42,
            bsWorkload=True,
            bsLoadShutdown=10,
            bsNumSamples=20,
            bsNumSteps=30,
            orderRate=0.5,
        )
        assert agent.bs_workload is True
        assert agent.bs_load_shutdown == 10
        assert agent.bs_n_samples == 20
        assert agent.bs_n_steps == 30
        assert agent.order_rate == 0.5
        agent = self._create_agent(seed=42, bsNumSamples=20, bsNumSteps=30)
        assert agent.bs_workload is False
        assert agent.bs_n_samples == 20
        assert agent.bs_n_steps == 30
        agent = self._create_agent(seed=42, orderRate=0)
        assert agent.order_rate == 0.0
        agent = self._create_agent(seed=42, orderRate=1)
        assert agent.order_rate == 1.0
        agent = self._create_agent(seed=42, orderRate={"const": [0.2]})
        assert agent.order_rate == 0.2
        agent = self._create_agent(seed=42, orderRate=[0.1, 0.2])
        assert 0.1 <= agent.order_rate < 0.2

    @pytest.mark.parametrize(
        "settings",
        [
            {"bsWorkload": "true"},
            {"bsWorkload": 1},
            {"bsLoadShutdown": 1.0},
            {"bsLoadShutdown": "1"},
            {"bsLoadShutdown": True},
            {"bsWorkload": True},
            {"bsWorkload": True, "bsNumSamples": 10},
            {"bsWorkload": True, "bsNumSteps": 10},
            {"bsNumSamples": 0},
            {"bsNumSamples": -1},
            {"bsNumSamples": 1.0},
            {"bsNumSamples": True},
            {"bsNumSteps": 0},
            {"bsNumSteps": "10"},
            {"orderRate": -0.1},
            {"orderRate": 1.1},
        ],
    )
    def test_setup_invalid(self, settings: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            self._create_agent(seed=42, **settings)

    @pytest.mark.parametrize(
        "load_shutdown, time, expected",
        [
            (-1, 0, True),
            (-1, 1000, True),
            (0, 1000, True),
            (1, 0, True),
            (1, 1, False),
            (10, 9, True),
            (10, 10, False),
            (10, 11, False),
        ],
    )
    def test_is_workload_on(
        self, load_shutdown: int, time: int, expected: bool
    ) -> None:
        agent = self._create_agent(
            seed=42,
            bsWorkload=True,
            bsLoadShutdown=load_shutdown,
            bsNumSamples=1,
            bsNumSteps=1,
        )
        assert agent.is_workload_on(time=time) is expected
        agent = self._create_agent(seed=42, bsLoadShutdown=load_shutdown)
        assert agent.is_workload_on(time=time) is False

    @pytest.mark.parametrize("seed", [1, 42])
    @pytest.mark.parametrize("load_shutdown, time", [(-1, 0), (5, 4), (5, 5)])
    def test_submit_orders(self, seed: int, load_shutdown: int, time: int) -> None:
        agent = self._create_agent(
            seed=seed,
            bsWorkload=True,
            bsLoadShutdown=load_shutdown,
            bsNumSamples=3,
            bsNumSteps=4,
            orderRate=0.5,
        )
        market = self._create_market(agent=agent, market_id=0, time=time)
        is_workload_on = agent.is_workload_on(time=time)
        prng = random.Random()
        prng.setstate(agent.prng.getstate())
        bs_sum = 0.0
        n_orders = 0
        for _ in range(100):
            if is_workload_on:
                black_scholes = _create_black_scholes(seed=0)
                black_scholes.prng = prng
                bs_sum += black_scholes.compute(n_samples=3, n_steps=4)
            is_ordered = prng.random() <= 0.5
            orders = agent.submit_orders(markets=[market])
            if is_ordered:
                # the gauss noise of FCNAgent
                prng.gauss(mu=0.0, sigma=1.0)
                assert len(orders) == 1
                assert isinstance(orders[0], Order)
                assert orders[0].market_id == 0
                n_orders += 1
            else:
                assert orders == []
            assert agent.prng.getstate() == prng.getstate()
        assert agent.bs_sum == bs_sum
        if is_workload_on:
            assert agent.bs_sum / 100 == pytest.approx(THEORETICAL_PRICE, abs=15.0)
        else:
            assert agent.bs_sum == 0.0
        assert 30 < n_orders < 70

    @pytest.mark.parametrize("order_rate, n_orders", [(0.0, 0), (1.0, 100)])
    def test_submit_orders_order_rate(self, order_rate: float, n_orders: int) -> None:
        agent = self._create_agent(seed=42, orderRate=order_rate)
        market = self._create_market(agent=agent, market_id=0)
        assert (
            sum(len(agent.submit_orders(markets=[market])) for _ in range(100))
            == n_orders
        )
        assert agent.bs_sum == 0.0

    def test_submit_orders_inaccessible(self) -> None:
        agent = self._create_agent(
            seed=42, bsWorkload=True, bsNumSamples=3, bsNumSteps=4, orderRate=1.0
        )
        market1 = self._create_market(agent=agent, market_id=1)
        market2 = self._create_market(agent=agent, market_id=2)
        prng_state = agent.prng.getstate()
        for _ in range(10):
            assert agent.submit_orders(markets=[]) == []
            assert agent.submit_orders(markets=[market1, market2]) == []
        # neither the workload is processed nor the random numbers are drawn
        assert agent.bs_sum == 0.0
        assert agent.prng.getstate() == prng_state

        # only the accessible market is ordered even if it is not the first one
        market0 = self._create_market(agent=agent, market_id=0)
        orders = agent.submit_orders(markets=[market1, market0, market2])
        assert len(orders) == 1
        assert orders[0].market_id == 0
        assert agent.bs_sum > 0.0


def _load_config() -> Dict[str, Any]:
    with open(
        os.path.join(SAMPLE_DIR, "config.json"), mode="r", encoding="utf-8"
    ) as fp:
        config: Dict[str, Any] = json.load(fp)
    return config


def _small_config() -> Dict[str, Any]:
    config = _load_config()
    config["simulation"]["sessions"][0]["iterationSteps"] = 6
    config["simulation"]["numParallel"] = 2
    config["FCNAgent"]["numAgents"] = 3
    config["FCNAgent"]["bsLoadShutdown"] = 3
    config["FCNAgent"]["bsNumSamples"] = 2
    config["FCNAgent"]["bsNumSteps"] = 3
    config["FCNAgent"]["orderRate"] = 0.5
    config["FCNAgents-I"]["numAgents"] = 2
    return config


def _agent_states(runner: SequentialRunner) -> List[Tuple[Any, ...]]:
    return [
        (
            agent.agent_id,
            agent.cash_amount,
            dict(agent.asset_volumes),
            agent.prng.getstate(),
        )
        for agent in runner.simulator.agents
    ]


def _bs_sums(runner: SequentialRunner) -> List[float]:
    return [
        agent.bs_sum
        for agent in runner.simulator.agents
        if isinstance(agent, WorkloadFCNAgent)
    ]


class TestWorkloadFCNAgentSimulation:
    def _run(
        self,
        config: Dict[str, Any],
        runner_class: Type[SequentialRunner] = SequentialRunner,
        seed: int = 42,
    ) -> Tuple[SequentialRunner, MarketStepSaver]:
        saver = MarketStepSaver()
        runner = runner_class(
            settings=copy.deepcopy(config), prng=random.Random(seed), logger=saver
        )
        runner.class_register(cls=WorkloadFCNAgent)
        runner.main()
        return runner, saver

    def test_config(self) -> None:
        runner, saver = self._run(config=_load_config())
        assert len(runner.simulator.markets) == 4
        assert len(runner.simulator.agents) == 71
        assert len(_bs_sums(runner=runner)) == 70
        assert all(bs_sum > 0.0 for bs_sum in _bs_sums(runner=runner))
        assert len(saver.market_step_logs) == 4 * 100
        # the fundamental price shock at the time 0 is applied only to SpotMarket-1
        assert runner.simulator.name2market[
            "SpotMarket-1"
        ].get_fundamental_price() == pytest.approx(270.0)
        assert runner.simulator.name2market[
            "SpotMarket-2"
        ].get_fundamental_price() == pytest.approx(300.0)

    @pytest.mark.parametrize("load_shutdown, n_workloads", [(-1, 6), (0, 6), (3, 3)])
    def test_load_shutdown(self, load_shutdown: int, n_workloads: int) -> None:
        config = _small_config()
        config["FCNAgent"]["bsLoadShutdown"] = load_shutdown
        with mock.patch.object(
            BlackScholes, "compute", autospec=True, return_value=1.0
        ) as compute:
            runner, _ = self._run(config=config)
        # every agent is asked to submit orders once in each step
        n_agents = len(_bs_sums(runner=runner))
        assert compute.call_count == n_workloads * n_agents
        assert _bs_sums(runner=runner) == [float(n_workloads)] * n_agents

    @pytest.mark.parametrize(
        "runner_class",
        [MultiThreadAgentParallelRunner, MultiProcessAgentParallelRunner],
    )
    def test_same_result_as_sequential(
        self, runner_class: Type[SequentialRunner]
    ) -> None:
        config = _small_config()
        sequential_runner, sequential_saver = self._run(config=config)
        parallel_runner, parallel_saver = self._run(
            config=config, runner_class=runner_class
        )
        assert parallel_saver.market_step_logs == sequential_saver.market_step_logs
        assert _agent_states(runner=parallel_runner) == _agent_states(
            runner=sequential_runner
        )
        assert all(bs_sum > 0.0 for bs_sum in _bs_sums(runner=sequential_runner))
        if runner_class is MultiProcessAgentParallelRunner:
            # bs_sum is updated only on the copies of the agents on the worker processes
            assert all(bs_sum == 0.0 for bs_sum in _bs_sums(runner=parallel_runner))
        else:
            assert _bs_sums(runner=parallel_runner) == _bs_sums(
                runner=sequential_runner
            )


def _run_main(argv: List[str], capsys: pytest.CaptureFixture) -> List[str]:
    capsys.readouterr()
    with mock.patch("sys.argv", ["main.py"] + argv):
        main()
    return [
        line
        for line in capsys.readouterr().out.splitlines()
        if not line.startswith("#")
    ]


def test_black_scholes() -> None:
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{SAMPLE_DIR}/config.json", "--seed", "1"]
    ):
        main()


def test_black_scholes_runners(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    assert list(RUNNERS.keys()) == ["sequential", "multi_thread", "multi_process"]
    config_path = str(tmp_path / "config.json")
    with open(config_path, mode="w", encoding="utf-8") as fp:
        json.dump(_small_config(), fp)
    sequential_lines = _run_main(
        argv=["--config", config_path, "--seed", "1"], capsys=capsys
    )
    assert len(sequential_lines) == 4 * 6
    assert (
        _run_main(
            argv=["-c", config_path, "-s", "1", "-r", "sequential"], capsys=capsys
        )
        == sequential_lines
    )
    for runner in ["multi_thread", "multi_process"]:
        assert (
            _run_main(
                argv=["--config", config_path, "--seed", "1", "--runner", runner],
                capsys=capsys,
            )
            == sequential_lines
        )
    with pytest.raises(SystemExit):
        _run_main(
            argv=["--config", config_path, "--runner", "multi_gpu"], capsys=capsys
        )
