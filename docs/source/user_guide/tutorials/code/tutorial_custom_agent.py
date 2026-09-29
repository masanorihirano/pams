"""Tutorial: write your own agents and use them in a simulation."""
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from pams.agents import Agent
from pams.agents import FCNAgent
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import OrderLog
from pams.market import Market
from pams.order import LIMIT_ORDER
from pams.order import Cancel
from pams.order import Order
from pams.runners import SequentialRunner
from pams.utils import JsonRandom


class MovingAverageAgent(Agent):
    """Trend follower comparing the market price with its moving average.

    When the market price is above the average of the last ``windowSize`` steps, the
    agent buys; when it is below, the agent sells. It keeps at most one order in each
    market: before placing a new order, it cancels its previous one if that order is
    not fully executed yet.
    """

    # [setup-start]
    window_size: int
    order_volume: int
    last_orders: Dict[int, Order]  # market ID -> last order placed
    n_submitted: int
    n_canceled: int
    trades: List[Tuple[int, float, int]]  # (step, price, +volume or -volume)

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Read the parameters of the agent from its block of the config.

        Args:
            settings (Dict[str, Any]): the agent's block of the config. It must include
                "windowSize" and can include "orderVolume" (default: 1).
            accessible_markets_ids (List[int]): IDs of the markets the agent can trade.
            *args: not used.
            **kwargs: not used.

        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        json_random = JsonRandom(prng=self.prng)
        self.window_size = int(json_random.random(settings["windowSize"]))
        self.order_volume = int(json_random.random(settings.get("orderVolume", 1)))
        self.last_orders = {}
        self.n_submitted = 0
        self.n_canceled = 0
        self.trades = []

    # [setup-end]

    # [submit-start]
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """Decide the orders of this step.

        Args:
            markets (List[Market]): all the markets of the simulation.

        Returns:
            List[Union[Order, Cancel]]: the orders and cancel orders to submit.

        """
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            if not self.is_market_accessible(market_id=market.market_id):
                continue
            time = market.get_time()
            if time < self.window_size:
                continue
            last_order = self.last_orders.pop(market.market_id, None)
            if last_order is not None and last_order.volume > 0:
                # not fully executed: without ttl, it is still in the order book
                orders.append(Cancel(order=last_order))
            past_prices = market.get_market_prices(
                times=range(time - self.window_size, time)
            )
            moving_average = sum(past_prices) / len(past_prices)
            price = market.get_market_price()
            if price > moving_average:
                is_buy = True
            elif price < moving_average:
                is_buy = False
            else:
                continue
            tick_level = market.convert_to_tick_level(price=price, is_buy=is_buy)
            order = Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=is_buy,
                kind=LIMIT_ORDER,
                volume=self.order_volume,
                price=market.convert_to_price(tick_level=tick_level),
            )
            orders.append(order)
            self.last_orders[market.market_id] = order
        return orders

    # [submit-end]

    # [callbacks-start]
    def submitted_order(self, log: OrderLog) -> None:
        """Count the orders that a market has accepted.

        Args:
            log (OrderLog): the accepted order.

        """
        self.n_submitted += 1

    def executed_order(self, log: ExecutionLog) -> None:
        """Record an execution of one of the agent's orders.

        Args:
            log (ExecutionLog): the execution.

        """
        volume = log.volume if log.buy_agent_id == self.agent_id else -log.volume
        self.trades.append((log.time, log.price, volume))

    def canceled_order(self, log: CancelLog) -> None:
        """Count the cancel orders that a market has accepted.

        Args:
            log (CancelLog): the canceled order.

        """
        self.n_canceled += 1

    # [callbacks-end]


# [fcn-start]
class ContrarianFCNAgent(FCNAgent):
    """FCN agent whose chart term is reversed: it expects past trends to revert."""

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Set up the FCN agent, then reverse its chart term.

        Args:
            settings (Dict[str, Any]): the agent's block of the config.
            accessible_markets_ids (List[int]): IDs of the markets the agent can trade.
            *args: not used.
            **kwargs: not used.

        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        self.is_chart_following = False


# [fcn-end]

# [config-start]
CONFIG: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents", "ContrarianAgents", "MovingAverageAgents"],
        "sessions": [
            {
                "sessionName": "warmup",
                "iterationSteps": 100,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
            },
            {
                "sessionName": "main",
                "iterationSteps": 500,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": True,
            },
        ],
    },
    "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents": 100,
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
    "ContrarianAgents": {
        "extends": "FCNAgents",
        "class": "ContrarianFCNAgent",
        "numAgents": 20,
        "chartWeight": {"expon": [1.0]},
    },
    "MovingAverageAgents": {
        "class": "MovingAverageAgent",
        "numAgents": 10,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "windowSize": [10, 30],
        "orderVolume": 1,
    },
}
# [config-end]


# [run-start]
def run_simulation(seed: int) -> SequentialRunner:
    """Register the user-defined agents and run the simulation.

    Args:
        seed (int): seed of the random number generator.

    Returns:
        SequentialRunner: the runner after the run.

    """
    runner = SequentialRunner(settings=CONFIG, prng=random.Random(seed))
    runner.class_register(cls=MovingAverageAgent)
    runner.class_register(cls=ContrarianFCNAgent)
    runner.main()
    return runner


def report(runner: SequentialRunner) -> None:
    """Print what each moving average agent did.

    Args:
        runner (SequentialRunner): the runner after the run.

    """
    market = runner.simulator.name2market["Market"]
    for agent in runner.simulator.agents_group_name2agent["MovingAverageAgents"]:
        if isinstance(agent, MovingAverageAgent):
            print(
                f"{agent.name}: window {agent.window_size}, "
                f"orders {agent.n_submitted}, cancels {agent.n_canceled}, "
                f"trades {len(agent.trades)}, "
                f"shares {agent.get_asset_volume(market_id=market.market_id)}, "
                f"cash {agent.get_cash_amount():.2f}"
            )


# [run-end]


def main() -> None:
    """Run the simulation with seed 42 and print the report."""
    report(run_simulation(seed=42))


if __name__ == "__main__":
    main()
