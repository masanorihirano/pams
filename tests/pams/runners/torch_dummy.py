import math
import random
import types
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from pams import LIMIT_ORDER
from pams import Simulator
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Cancel
from pams.order import Order

# PyTorch is imported in the functions and methods below so that this module can be imported
# without PyTorch. They are defined at the top level so that the worker processes can import them.

N_FEATURES = 4


class TorchPricingAgent(Agent):
    """Agent that places a limit order at the price predicted by a tiny PyTorch model.

    The model maps the last log returns of the market price to whether to buy and to the log ratio
    of the order price to the market price. The weights are initialized and the noise is drawn by
    PyTorch generators seeded by the agent's prng, so that the agent is deterministic for a given
    seed on any worker process.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent without the model, which is created by setup."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.model: Any = None

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # PyTorch is optional, so it is not imported at the top level
        import torch  # pylint: disable=import-outside-toplevel

        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        generator = torch.Generator().manual_seed(self.prng.randrange(2**32))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
        )
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        # PyTorch is optional, so it is not imported at the top level
        import torch  # pylint: disable=import-outside-toplevel

        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            if not self.is_market_accessible(market_id=market.market_id):
                continue
            time = market.get_time()
            times = range(max(time - N_FEATURES, 0), time + 1)
            prices = market.get_market_prices(times)
            prices = [prices[0]] * (N_FEATURES + 1 - len(prices)) + prices
            log_returns = [
                100 * math.log(price / previous_price)
                for previous_price, price in zip(prices[:-1], prices[1:])
            ]
            generator = torch.Generator().manual_seed(self.prng.randrange(2**32))
            with torch.no_grad():
                output = self.model(torch.tensor(log_returns)) + 0.1 * torch.randn(
                    2, generator=generator
                )
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=bool(output[0] > 0),
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=market.get_market_price()
                    * math.exp(0.01 * math.tanh(float(output[1]))),
                    ttl=10,
                )
            )
        return orders


def get_torch_settings() -> Tuple[int, str]:
    """Get the number of threads and the sharing strategy of PyTorch on the current process."""
    # PyTorch is optional, so it is not imported at the top level
    import torch.multiprocessing  # pylint: disable=import-outside-toplevel

    return torch.get_num_threads(), torch.multiprocessing.get_sharing_strategy()


def inspect_tensor(tensor: Any) -> Tuple[bool, float]:
    """Get whether the tensor is in shared memory and its sum on the current process."""
    return tensor.is_shared(), float(tensor.sum())


class FakeTorchMultiprocessing(types.ModuleType):
    """Fake :mod:`torch.multiprocessing` with only the functions used by the runner."""

    def __init__(self) -> None:
        """Initialize the module with the default sharing strategy of Linux."""
        super().__init__("torch.multiprocessing")
        self.sharing_strategy: str = "file_descriptor"

    def get_sharing_strategy(self) -> str:
        return self.sharing_strategy

    def set_sharing_strategy(self, new_strategy: str) -> None:
        self.sharing_strategy = new_strategy


class FakeTorch(types.ModuleType):
    """Fake :mod:`torch` with only the functions used by the runner.

    It lets the tests check the logic of the runner on the main process without PyTorch.
    """

    def __init__(self, num_threads: int) -> None:
        """Initialize the module with the given number of threads."""
        super().__init__("torch")
        self.num_threads: int = num_threads
        self.multiprocessing: FakeTorchMultiprocessing = FakeTorchMultiprocessing()

    def get_num_threads(self) -> int:
        return self.num_threads

    def set_num_threads(self, num_threads: int) -> None:
        self.num_threads = num_threads
