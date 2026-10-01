"""Tutorial: write your own logger to record the trades of a simulation."""
import csv
import random
from typing import Any
from typing import Dict
from typing import List

from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import OrderLog
from pams.logs import SessionBeginLog
from pams.logs import SessionEndLog
from pams.logs import SimulationEndLog
from pams.runners import SequentialRunner


# [logger-start]
class TradeLogger(Logger):
    """Print a summary of each session and save every execution to a CSV file."""

    def __init__(self, path: str) -> None:
        """Initialize the logger.

        Args:
            path (str): path of the CSV file written at the end of the simulation.

        """
        super().__init__()
        self.path: str = path
        self.executions: List[ExecutionLog] = []
        self.n_orders: int = 0
        self.volume: int = 0

    def process_session_begin_log(self, log: SessionBeginLog) -> None:
        """Reset the counters at the beginning of each session.

        Args:
            log (SessionBeginLog): the log of the session that begins.

        """
        self.n_orders = 0
        self.volume = 0

    def process_order_log(self, log: OrderLog) -> None:
        """Count the orders accepted by the markets.

        Args:
            log (OrderLog): the accepted order.

        """
        self.n_orders += 1

    def process_execution_log(self, log: ExecutionLog) -> None:
        """Keep the execution and add its volume to the volume of the session.

        Args:
            log (ExecutionLog): the execution.

        """
        self.executions.append(log)
        self.volume += log.volume

    def process_session_end_log(self, log: SessionEndLog) -> None:
        """Print the counters at the end of each session.

        Args:
            log (SessionEndLog): the log of the session that ends.

        """
        print(
            f"{log.session.name}: {self.n_orders} orders, "
            f"{self.volume} shares traded"
        )

    def process_simulation_end_log(self, log: SimulationEndLog) -> None:
        """Write all the executions to the CSV file.

        Args:
            log (SimulationEndLog): the log of the end of the simulation.

        """
        with open(self.path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["time", "market", "price", "volume", "buyer", "seller"])
            for execution in self.executions:
                writer.writerow(
                    [
                        execution.time,
                        self.simulator.id2market[execution.market_id].name,
                        execution.price,
                        execution.volume,
                        self.simulator.id2agent[execution.buy_agent_id].name,
                        self.simulator.id2agent[execution.sell_agent_id].name,
                    ]
                )


# [logger-end]

# [config-start]
CONFIG: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents"],
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
}
# [config-end]


# [run-start]
def run_simulation(seed: int, path: str) -> SequentialRunner:
    """Run the simulation with the trade logger.

    Args:
        seed (int): seed of the random number generator.
        path (str): path of the CSV file to write.

    Returns:
        SequentialRunner: the runner after the run.

    """
    runner = SequentialRunner(
        settings=CONFIG, prng=random.Random(seed), logger=TradeLogger(path=path)
    )
    runner.main()
    return runner


# [run-end]


def main() -> None:
    """Run the simulation with seed 42 and print the first lines of the CSV file."""
    run_simulation(seed=42, path="trades.csv")
    with open("trades.csv", encoding="utf-8") as fp:
        for line in fp.readlines()[:4]:
            print(line.rstrip())


if __name__ == "__main__":
    main()
