"""Agents using TensorFlow for the tests of TensorFlowAgentParallelRunner.

TensorFlow is imported in the functions by _import_tensorflow, not at the top level, so that this
module can be imported without TensorFlow. The functions are defined at the top level so that the
worker processes can import them.
"""
import functools
import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from pams import LIMIT_ORDER
from pams import Simulator
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Cancel
from pams.order import Order
from pams.runners import SequentialRunner
from pams.runners.tensorflow_parallel import _import_tensorflow

from .dummy import DummyLogger2

N_RETURNS = 3
N_FEATURES = N_RETURNS + 1
N_HIDDEN = 8
# shape of the samples of TensorFlowReductionAgent, large enough to be reduced on several threads
N_SAMPLES_SHAPE = (2048, 512)


@functools.lru_cache(maxsize=None)
def get_price_model(seed: int) -> Any:
    """Get a small Keras model mapping market features to the expected log return.

    The weights are drawn by numpy from the seed, so the model is the same on every process.
    The model is cached, so each process builds it only once.
    """
    tf = _import_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(N_FEATURES,)),
            tf.keras.layers.Dense(N_HIDDEN, activation="tanh"),
            tf.keras.layers.Dense(1, activation="tanh"),
        ]
    )
    rng = np.random.default_rng(seed)
    model.set_weights(
        [
            rng.normal(size=weight.shape).astype(np.float32)
            for weight in model.get_weights()
        ]
    )
    return model


def load_price_model(seed: int) -> None:
    """Load the price model into the cache of this module, as a worker initializer.

    The model is also called once, so TensorFlow is initialized on the worker process.
    """
    get_price_model(seed)(np.zeros((1, N_FEATURES), dtype=np.float32), training=False)


def get_n_cached_price_models() -> int:
    """Get the number of the price models cached on the current process."""
    return get_price_model.cache_info().currsize


def get_features(market: Market) -> List[float]:
    """Get the features of the market in percent.

    The features are the last log returns of the market price and the log ratio of the
    fundamental price to the market price.
    """
    time = market.get_time()
    prices = market.get_market_prices(range(max(time - N_RETURNS, 0), time + 1))
    prices = [prices[0]] * (N_RETURNS + 1 - len(prices)) + prices
    log_returns = [math.log(prices[i + 1] / prices[i]) for i in range(N_RETURNS)]
    log_ratio = math.log(market.get_fundamental_price() / market.get_market_price())
    return [100.0 * value for value in log_returns + [log_ratio]]


def get_tensorflow_config() -> Tuple[int, int, List[bool]]:
    """Get the numbers of intra-op and inter-op threads and the memory growth of the GPUs."""
    tf = _import_tensorflow()
    return (
        tf.config.threading.get_intra_op_parallelism_threads(),
        tf.config.threading.get_inter_op_parallelism_threads(),
        [
            tf.config.experimental.get_memory_growth(gpu)
            for gpu in tf.config.list_physical_devices("GPU")
        ],
    )


class TensorFlowAgent(Agent):
    """Agent that submits a limit order at the price predicted by a small TensorFlow model.

    The agent holds only the seed of the model and gets the model from the cache of this module,
    as recommended for TensorFlowAgentParallelRunner. The noise is drawn from the agent's prng.
    The agent submits an order to every market, so it must be able to access all the markets.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent with placeholders of the settings, which are set by setup."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.model_seed: int = 0
        self.noise_scale: float = 0.0

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.model_seed = settings["modelSeed"]
        self.noise_scale = settings["noiseScale"]

    def get_model(self) -> Any:
        return get_price_model(self.model_seed)

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        model = self.get_model()
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            features = np.asarray([get_features(market=market)], dtype=np.float32)
            expected_log_return = 0.001 * float(
                model(features, training=False).numpy()[0, 0]
            )
            noise = self.noise_scale * self.prng.gauss(0.0, 1.0)
            market_price = market.get_market_price()
            order_price = market_price * math.exp(expected_log_return + noise)
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=order_price > market_price,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=order_price,
                    ttl=20,
                )
            )
        return orders


class TensorFlowModelHoldingAgent(TensorFlowAgent):
    """TensorFlowAgent that holds its Keras model as an attribute.

    The model is pickled in every task of TensorFlowAgentParallelRunner, which is slow but must
    give the same results.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent without the model, which is built by setup."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.model: Any = None

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.model = get_price_model(self.model_seed)

    def get_model(self) -> Any:
        return self.model


class TensorFlowReductionAgent(TensorFlowAgent):
    """TensorFlowAgent whose noise is the mean of about a million samples drawn by TensorFlow.

    The samples are drawn by a stateless random operation seeded from the agent's prng, so the
    agent is deterministic. However, TensorFlow computes the mean on several threads, and its
    rounding errors depend on the number of intra-op threads.
    """

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        tf = _import_tensorflow()
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            seed = [self.prng.randrange(2**31), self.prng.randrange(2**31)]
            samples = tf.random.stateless_normal(N_SAMPLES_SHAPE, seed=seed)
            # the mean of the samples is multiplied by the square root of their number
            # so that the noise follows the standard normal distribution
            noise = float(tf.reduce_mean(samples).numpy()) * math.sqrt(
                math.prod(N_SAMPLES_SHAPE)
            )
            market_price = market.get_market_price()
            order_price = market_price * math.exp(self.noise_scale * noise)
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=order_price > market_price,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=order_price,
                    ttl=20,
                )
            )
        return orders


def run_sequential_runner(setting: Dict, seed: int) -> SequentialRunner:
    """Run SequentialRunner with the agents of this module on the current process.

    This is called on a new process to run SequentialRunner with TensorFlow configured in the
    same way as the worker processes of TensorFlowAgentParallelRunner.
    """
    runner = SequentialRunner(
        settings=setting, prng=random.Random(seed), logger=DummyLogger2()
    )
    runner.class_register(cls=TensorFlowAgent)
    runner.class_register(cls=TensorFlowModelHoldingAgent)
    runner.class_register(cls=TensorFlowReductionAgent)
    runner._setup()
    runner._run()
    return runner
