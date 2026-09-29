"""Agents and worker functions using JAX and Flax for the tests of JaxAgentParallelRunner.

This module imports JAX and Flax, so import it only when they are installed.
The functions are defined at the top level so that the runners can pickle them and call them on
worker processes.
"""
import math
import os
import random
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from pams import LIMIT_ORDER
from pams import Simulator
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Cancel
from pams.order import Order


class ProcessState:
    """State of the current process, updated by the functions below."""

    def __init__(self) -> None:
        """Initialize the state of a process that has not run any function below."""
        # number of times that predict_price_change is traced, i.e., compiled
        self.trace_count: int = 0
        # parameters of the model loaded by load_shared_model
        self.shared_params: Optional[Any] = None
        # XLA_PYTHON_CLIENT_PREALLOCATE when load_shared_model was called
        self.preallocate_at_load: Optional[str] = None


PROCESS_STATE = ProcessState()


class PriceModel(nn.Module):
    """Tiny MLP mapping market features to the relative change of the order price in (-1, 1)."""

    hidden_size: int = 4

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        hidden = nn.tanh(nn.Dense(self.hidden_size)(features))
        return jnp.tanh(nn.Dense(1)(hidden)[..., 0])


N_FEATURES = 3


def init_params(model: PriceModel, seed: int) -> Any:
    """Initialize the parameters of the model with the seed."""
    return model.init(jax.random.PRNGKey(seed), jnp.zeros((N_FEATURES,), jnp.float32))


def _predict_price_change(
    model: PriceModel, params: Any, features: jax.Array
) -> jax.Array:
    # this line runs only when the function is traced
    PROCESS_STATE.trace_count += 1
    return model.apply(params, features)


# the model is a static argument, so the copies of the model pickled in the tasks hit the cache
predict_price_change = jax.jit(_predict_price_change, static_argnums=0)


def get_trace_count() -> int:
    """Get the number of times that predict_price_change is traced on the current process."""
    return PROCESS_STATE.trace_count


def load_shared_model(seed: int) -> None:
    """Load the parameters of the shared model on the current process, as a worker initializer."""
    PROCESS_STATE.preallocate_at_load = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    PROCESS_STATE.shared_params = init_params(model=PriceModel(), seed=seed)


def get_jax_worker_config() -> Dict[str, Optional[str]]:
    """Get the configuration of JAX on the current process."""
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION"
        ),
        "XLA_CLIENT_MEM_FRACTION": os.environ.get("XLA_CLIENT_MEM_FRACTION"),
        "jax_platforms": jax.config.jax_platforms,
        "default_backend": jax.default_backend(),
        "preallocate_at_load": PROCESS_STATE.preallocate_at_load,
    }


def get_device_count() -> int:
    """Get the number of the devices of JAX on the current process."""
    return jax.device_count()


def _market_features(market: Market, noise: float) -> jax.Array:
    market_price = market.get_market_price()
    past_price = market.get_market_price(max(market.get_time() - 5, 0))
    return jnp.asarray(
        [
            math.log(market.get_fundamental_price() / market_price) * 10.0,
            math.log(market_price / past_price) * 10.0,
            noise,
        ],
        dtype=jnp.float32,
    )


class PriceModelAgent(Agent):
    """Agent submitting a limit order whose price is decided by a tiny Flax model.

    The random numbers are derived from ``self.prng`` so that the results do not depend on the
    runner.
    """

    @abstractmethod
    def _get_model_and_params(self) -> Tuple[PriceModel, Any]:
        """Get the model and its parameters used by submit_orders."""

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        model, params = self._get_model_and_params()
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            if not self.is_market_accessible(market_id=market.market_id):
                continue
            features = _market_features(market=market, noise=self.prng.gauss(0.0, 1.0))
            price_change = float(predict_price_change(model, params, features))
            price = market.get_market_price() * (1.0 + 0.01 * price_change)
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=self.prng.random() < 0.5,
                    kind=LIMIT_ORDER,
                    volume=1,
                    price=round(price / market.tick_size) * market.tick_size,
                    ttl=10,
                )
            )
        return orders


class FlaxPriceAgent(PriceModelAgent):
    """PriceModelAgent with its own model.

    The model and its parameters are attributes of the agent, so they are pickled with the agent in
    every task of the process runners. The parameters are initialized with a seed from
    ``self.prng``.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the agent without the parameters, which are initialized by setup."""
        super().__init__(agent_id, prng, simulator, name, logger)
        self.model = PriceModel()
        self.params: Any = None

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.params = init_params(model=self.model, seed=self.prng.randrange(2**31))

    def _get_model_and_params(self) -> Tuple[PriceModel, Any]:
        return self.model, self.params


class SharedFlaxModelAgent(PriceModelAgent):
    """PriceModelAgent using the model loaded by load_shared_model on the current process.

    The model is not pickled in the tasks.
    """

    def _get_model_and_params(self) -> Tuple[PriceModel, Any]:
        if PROCESS_STATE.shared_params is None:
            raise RuntimeError("the shared model is not loaded on this process")
        return PriceModel(), PROCESS_STATE.shared_params
