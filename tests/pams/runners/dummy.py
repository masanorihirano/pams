import time
from typing import List
from typing import Union

from pams.agents.fcn_agent import FCNAgent
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import MarketStepBeginLog
from pams.logs import MarketStepEndLog
from pams.logs import OrderLog
from pams.logs import SessionBeginLog
from pams.logs import SessionEndLog
from pams.logs import SimulationBeginLog
from pams.logs import SimulationEndLog
from pams.market import Market
from pams.order import Cancel
from pams.order import Order

wait_time = 0.2  # seconds


class FCNDelayAgent(FCNAgent):
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        time.sleep(wait_time)  # Simulate a delay
        return super().submit_orders(markets)


class DummyLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.n_market_step_begin = 0
        self.n_market_end_begin = 0

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        self.n_market_step_begin += 1

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        self.n_market_end_begin += 1


class DummyLogger2(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.n_order_log = 0
        self.n_cancel_log = 0
        self.n_execution_log = 0
        self.n_simulation_begin_log = 0
        self.n_simulation_end_log = 0
        self.n_session_begin_log = 0
        self.n_session_end_log = 0
        self.n_market_step_begin = 0
        self.n_market_step_end = 0

    def process_order_log(self, log: OrderLog) -> None:
        self.n_order_log += 1

    def process_cancel_log(self, log: CancelLog) -> None:
        self.n_cancel_log += 1

    def process_execution_log(self, log: ExecutionLog) -> None:
        self.n_execution_log += 1

    def process_simulation_begin_log(self, log: SimulationBeginLog) -> None:
        self.n_simulation_begin_log += 1

    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        self.n_simulation_end_log += 1

    def process_session_begin_log(self, log: SessionBeginLog) -> None:
        self.n_session_begin_log += 1

    def process_session_end_log(self, log: SessionEndLog) -> None:
        self.n_session_end_log += 1

    def process_market_step_begin_log(self, log: MarketStepBeginLog) -> None:
        self.n_market_step_begin += 1

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        self.n_market_step_end += 1
