import time
from typing import List, Union

from pams.agents.fcn_agent import FCNAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order

wait_time = 0.2  # seconds


class FCNDelayAgent(FCNAgent):
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        time.sleep(wait_time)  # Simulate a delay
        return super().submit_orders(markets)
