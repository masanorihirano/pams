"""Agents using TensorFlow for the tests of TensorFlowAgentParallelRunner.

TensorFlow is imported in the functions, not at the top level, so that this module can be imported
without TensorFlow. The functions are defined at the top level so that the worker processes can
import them.
"""
import functools
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from pams import LIMIT_ORDER
from pams.agents import Agent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order

N_RETURNS = 3
N_FEATURES = N_RETURNS + 1
N_HIDDEN = 8


@functools.lru_cache(maxsize=None)
def get_price_model(seed: int) -> Any:
    """Get a small Keras model mapping market features to the expected log return.

    The weights are drawn by numpy from the seed, so the model is the same on every process.
    The model is cached, so each process builds it only once.
    """
    import tensorflow as tf

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
    import tensorflow as tf

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
    """

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.model_seed: int = settings["modelSeed"]
        self.noise_scale: float = settings["noiseScale"]

    def get_model(self) -> Any:
        return get_price_model(self.model_seed)

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        model = self.get_model()
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            if not self.is_market_accessible(market_id=market.market_id):
                continue
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

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.model: Any = get_price_model(self.model_seed)

    def get_model(self) -> Any:
        return self.model
