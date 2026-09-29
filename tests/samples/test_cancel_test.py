import copy
import os.path
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from unittest import mock

import pytest

from pams import LIMIT_ORDER
from pams import Cancel
from pams import Market
from pams import Order
from pams import Simulator
from pams.agents import FCNAgent
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import OrderLog
from pams.logs import SimulationEndLog
from pams.runners import SequentialRunner
from samples.cancel_test.cancel_fcn_agent import DEFAULT_CANCEL_RATE
from samples.cancel_test.cancel_fcn_agent import CancelFCNAgent
from samples.cancel_test.main import main


class OrderAndCancelLogger(Logger):
    def __init__(self) -> None:
        """Initialize the log lists."""
        super().__init__()
        self.order_logs: List[OrderLog] = []
        self.cancel_logs: List[CancelLog] = []
        self.execution_logs: List[ExecutionLog] = []
        self.book_orders: List[Order] = []

    def process_order_log(self, log: OrderLog) -> None:
        self.order_logs.append(log)

    def process_cancel_log(self, log: CancelLog) -> None:
        self.cancel_logs.append(log)

    def process_execution_log(self, log: ExecutionLog) -> None:
        self.execution_logs.append(log)

    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        for market in log.simulator.markets:
            self.book_orders.extend(market.buy_order_book.priority_queue)
            self.book_orders.extend(market.sell_order_book.priority_queue)


class TestCancelFCNAgent:
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

    def _create_agent(self, seed: int, cancel_rate: Any = None) -> CancelFCNAgent:
        sim = Simulator(prng=random.Random(seed + 1))
        agent = CancelFCNAgent(
            agent_id=1, prng=random.Random(seed), simulator=sim, name="test_agent"
        )
        settings = copy.deepcopy(self.settings)
        if cancel_rate is not None:
            settings["cancelRate"] = cancel_rate
        agent.setup(settings=settings, accessible_markets_ids=[0, 1])
        return agent

    def _create_market(self, agent: CancelFCNAgent, market_id: int) -> Market:
        market = Market(
            market_id=market_id,
            prng=random.Random(market_id),
            simulator=agent.simulator,
            name=f"market{market_id}",
        )
        market._update_time(next_fundamental_price=300.0)
        return market

    def test_setup(self) -> None:
        agent = self._create_agent(seed=42)
        assert agent.cancel_rate == DEFAULT_CANCEL_RATE == 0.3
        assert agent.time_window_size == 100
        assert agent.order_margin == 0.1

        agent = self._create_agent(seed=42, cancel_rate=0.5)
        assert agent.cancel_rate == 0.5
        agent = self._create_agent(seed=42, cancel_rate=0)
        assert agent.cancel_rate == 0.0
        agent = self._create_agent(seed=42, cancel_rate=1)
        assert agent.cancel_rate == 1.0
        agent = self._create_agent(seed=42, cancel_rate={"const": [0.2]})
        assert agent.cancel_rate == 0.2
        agent = self._create_agent(seed=42, cancel_rate=[0.1, 0.2])
        assert 0.1 <= agent.cancel_rate < 0.2

        with pytest.raises(ValueError):
            self._create_agent(seed=42, cancel_rate=-0.1)
        with pytest.raises(ValueError):
            self._create_agent(seed=42, cancel_rate=1.1)

    @pytest.mark.parametrize("seed", [1, 42, 100, 200])
    def test_submit_orders(self, seed: int) -> None:
        agent = self._create_agent(seed=seed)
        market = self._create_market(agent=agent, market_id=0)
        _prng = random.Random(seed)
        n_cancels = 0
        for _ in range(1000):
            orders = agent.submit_orders(markets=[market])
            # the gauss noise of FCNAgent and then one draw per order
            _prng.gauss(mu=0.0, sigma=1.0)
            is_canceled = _prng.random() < 0.3
            assert isinstance(orders[0], Order)
            if is_canceled:
                assert len(orders) == 2
                cancel = orders[1]
                assert isinstance(cancel, Cancel)
                assert cancel.order is orders[0]
                assert cancel.placed_at is None
                n_cancels += 1
            else:
                assert len(orders) == 1
        assert 250 < n_cancels < 350

    @pytest.mark.parametrize("seed", [1, 42])
    def test_submit_orders_cancel_rate(self, seed: int) -> None:
        agent = self._create_agent(seed=seed, cancel_rate=0.0)
        market = self._create_market(agent=agent, market_id=0)
        for _ in range(100):
            orders = agent.submit_orders(markets=[market])
            assert len(orders) == 1
            assert isinstance(orders[0], Order)

        agent = self._create_agent(seed=seed, cancel_rate=1.0)
        market0 = self._create_market(agent=agent, market_id=0)
        market1 = self._create_market(agent=agent, market_id=1)
        market2 = self._create_market(agent=agent, market_id=2)
        for _ in range(100):
            orders = agent.submit_orders(markets=[market0, market1, market2])
            # the cancel orders follow the orders of each market and
            # no order is submitted to the inaccessible market
            assert len(orders) == 4
            order0, cancel0, order1, cancel1 = orders
            assert isinstance(order0, Order)
            assert isinstance(order1, Order)
            assert order0.market_id == 0
            assert order1.market_id == 1
            assert isinstance(cancel0, Cancel)
            assert isinstance(cancel1, Cancel)
            assert cancel0.order is order0
            assert cancel1.order is order1

        prng_state = agent.prng.getstate()
        assert not agent.submit_orders_by_market(market=market2)
        # no random number is drawn when no order is made
        assert agent.prng.getstate() == prng_state

    def test_submit_orders_by_market_multiple_orders(self) -> None:
        agent = self._create_agent(seed=42, cancel_rate=1.0)
        market = self._create_market(agent=agent, market_id=0)
        order0 = Order(
            agent_id=1,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=290.0,
        )
        order1 = Order(
            agent_id=1,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            price=299.0,
        )
        order2 = Order(
            agent_id=1,
            market_id=0,
            is_buy=False,
            kind=LIMIT_ORDER,
            volume=1,
            price=301.0,
        )
        cancel = Cancel(order=order0)
        prng_state = agent.prng.getstate()
        with mock.patch.object(
            FCNAgent, "submit_orders_by_market", return_value=[order1, cancel, order2]
        ):
            orders = agent.submit_orders_by_market(market=market)
        # the cancel orders follow all the orders and
        # a cancel order made by FCNAgent is passed through as it is
        assert len(orders) == 5
        assert orders[0] is order1
        assert orders[1] is cancel
        assert orders[2] is order2
        for cancel_, order in zip(orders[3:], [order1, order2]):
            assert isinstance(cancel_, Cancel)
            assert cancel_.order is order
        # one random number is drawn per order and none for the cancel order
        _prng = random.Random()
        _prng.setstate(prng_state)
        _prng.random()
        _prng.random()
        assert agent.prng.getstate() == _prng.getstate()


class TestCancelFCNAgentSimulation:
    config: Dict[str, Any] = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["CancelFCNAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 20,
                    "withOrderPlacement": True,
                    "withOrderExecution": False,
                    "withPrint": True,
                },
                {
                    "sessionName": 1,
                    "iterationSteps": 50,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                },
            ],
        },
        "Market": {
            "class": "Market",
            "tickSize": 0.00001,
            "marketPrice": 300.0,
            "outstandingShares": 25000,
        },
        "CancelFCNAgents": {
            "class": "CancelFCNAgent",
            "numAgents": 10,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": {"expon": [1.0]},
            "chartWeight": {"expon": [0.0]},
            "noiseWeight": {"expon": [1.0]},
            "noiseScale": 0.001,
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
        },
    }

    def _run(self, config: Dict[str, Any], seed: int) -> OrderAndCancelLogger:
        logger = OrderAndCancelLogger()
        runner = SequentialRunner(
            settings=config, prng=random.Random(seed), logger=logger
        )
        runner.class_register(cls=CancelFCNAgent)
        runner.main()
        return logger

    @pytest.mark.parametrize("seed", [1, 42])
    def test_cancel(self, seed: int) -> None:
        logger = self._run(config=self.config, seed=seed)
        order_logs: Dict[int, OrderLog] = {
            log.order_id: log for log in logger.order_logs
        }
        assert len(order_logs) == 70
        assert 0 < len(logger.cancel_logs) < len(order_logs)
        executed_volumes: Dict[int, int] = {}
        for execution_log in logger.execution_logs:
            for order_id in [execution_log.buy_order_id, execution_log.sell_order_id]:
                executed_volumes[order_id] = (
                    executed_volumes.get(order_id, 0) + execution_log.volume
                )
        n_executed_cancels = 0
        for cancel_log in logger.cancel_logs:
            order_log = order_logs[cancel_log.order_id]
            assert cancel_log.agent_id == order_log.agent_id
            # each order is canceled at the step when it is placed
            assert cancel_log.cancel_time == order_log.time == cancel_log.order_time
            executed_volume = executed_volumes.get(cancel_log.order_id, 0)
            if cancel_log.cancel_time < 20:
                # no order is executed in the first session
                assert executed_volume == 0
            # the cancel order only removes the volume that is not executed and
            # has no effect if the order is fully executed when it is placed
            assert cancel_log.volume == order_log.volume - executed_volume
            if cancel_log.volume == 0:
                n_executed_cancels += 1
        assert n_executed_cancels > 0
        canceled_ids = {log.order_id for log in logger.cancel_logs}
        assert len(logger.book_orders) > 0
        for order in logger.book_orders:
            assert not order.is_canceled
            assert order.order_id not in canceled_ids

    @pytest.mark.parametrize("seed", [1, 42])
    def test_cancel_all(self, seed: int) -> None:
        config = copy.deepcopy(self.config)
        config["simulation"]["sessions"] = config["simulation"]["sessions"][:1]
        config["CancelFCNAgents"]["cancelRate"] = 1.0
        logger = self._run(config=config, seed=seed)
        assert len(logger.order_logs) == 20
        assert [log.order_id for log in logger.cancel_logs] == [
            log.order_id for log in logger.order_logs
        ]
        assert not logger.book_orders

    @pytest.mark.parametrize("seed", [1, 42])
    def test_deterministic(self, seed: int) -> None:
        def _get_logs(
            logger: OrderAndCancelLogger,
        ) -> List[List[Union[int, float, None]]]:
            return [
                [log.order_id, log.agent_id, log.time, log.price]
                for log in logger.order_logs
            ] + [
                [log.order_id, log.agent_id, log.cancel_time, log.volume]
                for log in logger.cancel_logs
            ]

        logs1 = _get_logs(self._run(config=self.config, seed=seed))
        logs2 = _get_logs(self._run(config=self.config, seed=seed))
        assert logs1 == logs2


def test_cancel_test() -> None:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    sample_dir = os.path.join(root_dir, "samples", "cancel_test")
    with mock.patch(
        "sys.argv", ["main.py", "--config", f"{sample_dir}/config.json", "--seed", "1"]
    ):
        main()
