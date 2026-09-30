"""WorkloadFCNAgent, an FCN agent with an artificial computational workload.

This is the counterpart of WorkloadFCNAgent of the Parallel sample of Plham. The
workload is a Monte Carlo pricing of a European call option in the Black-Scholes model.
"""

import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents import FCNAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order
from pams.utils.json_random import JsonRandom

# ORDER_RATE of run.sh of the Parallel sample of Plham
DEFAULT_ORDER_RATE: float = 0.1

# the option priced by the workload of WorkloadFCNAgent of Plham
BS_INITIAL_PRICE: float = 100.0
BS_STRIKE_PRICE: float = 100.0
BS_RISK_FREE_RATE: float = 0.1
BS_VOLATILITY: float = 0.3
BS_MATURITY_TIME: float = 3.0


def black_scholes_monte_carlo(
    prng: random.Random,
    n_samples: int,
    n_steps: int,
    initial_price: float = BS_INITIAL_PRICE,
    strike_price: float = BS_STRIKE_PRICE,
    risk_free_rate: float = BS_RISK_FREE_RATE,
    volatility: float = BS_VOLATILITY,
    maturity_time: float = BS_MATURITY_TIME,
) -> float:
    """Price a European call option by a Monte Carlo simulation.

    The price of the underlying asset follows the Black-Scholes model and is simulated
    by the Euler method with ``n_steps`` steps, as plham.util.BlackScholes of Plham
    does. The price of the option is the discounted mean of the payoffs of
    ``n_samples`` paths.

    Args:
        prng (random.Random): pseudo random number generator for the paths.
        n_samples (int): number of paths.
        n_steps (int): number of time steps of each path.
        initial_price (float): initial price of the underlying asset.
        strike_price (float): strike price of the option.
        risk_free_rate (float): risk-free interest rate.
        volatility (float): volatility of the underlying asset.
        maturity_time (float): time to maturity.

    Returns:
        float: estimated price of the option.

    """
    if n_samples < 1:
        raise ValueError("n_samples have to be positive")
    if n_steps < 1:
        raise ValueError("n_steps have to be positive")
    dt: float = maturity_time / n_steps
    sqrt_dt: float = math.sqrt(dt)
    total: float = 0.0
    for _ in range(n_samples):
        price: float = initial_price
        for _ in range(n_steps):
            price += (
                price * risk_free_rate * dt
                + price * prng.gauss(0.0, 1.0) * volatility * sqrt_dt
            )
        total += max(price - strike_price, 0.0)
    return math.exp(-risk_free_rate * maturity_time) * total / n_samples


def _read_positive_int(settings: Dict[str, Any], key: str) -> int:
    """Read a positive integer from the settings (internal usage).

    Args:
        settings (Dict[str, Any]): agent configuration.
        key (str): key of the parameter.

    Returns:
        int: value of the parameter.

    """
    if key not in settings:
        raise ValueError(
            f"{key} is required for WorkloadFCNAgent if bsWorkload is true"
        )
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} have to be a positive int")
    return value


class WorkloadFCNAgent(FCNAgent):
    """Workload FCN Agent class.

    This agent adds a computational workload to the FCN agent, as WorkloadFCNAgent of
    the Parallel sample of Plham does. Every time it is asked to submit orders to an
    accessible market while the workload is on, it prices an option by
    :func:`black_scholes_monte_carlo` and discards the price. Then, it submits orders
    in the same way as FCN agents with the probability ``order_rate``.
    This class inherits from the :class:`pams.agents.FCNAgent` class.

    The Monte Carlo simulation draws random numbers from its own pseudo random number
    generator (see :func:`compute_workload`), so the workload changes neither the
    orders nor the results of the simulation.
    """

    order_rate: float
    bs_workload: bool
    bs_n_samples: int
    bs_n_steps: int

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Agent setup.  Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. This can include the
                parameters "orderRate" (0.1 if not specified), "bsWorkload" (false if
                not specified), "bsNumSamples", and "bsNumSteps" in addition to the
                parameters of :class:`pams.agents.FCNAgent`. "bsNumSamples" and
                "bsNumSteps" are required if "bsWorkload" is true.
            accessible_markets_ids (List[int]): list of market IDs.
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        if "orderRate" in settings:
            json_random: JsonRandom = JsonRandom(prng=self.prng)
            self.order_rate = json_random.random(json_value=settings["orderRate"])
        else:
            self.order_rate = DEFAULT_ORDER_RATE
        if not 0.0 <= self.order_rate <= 1.0:
            raise ValueError("orderRate have to be between 0.0 and 1.0")
        self.bs_workload = settings.get("bsWorkload", False)
        if not isinstance(self.bs_workload, bool):
            raise ValueError("bsWorkload have to be bool")
        self.bs_n_samples = 0
        self.bs_n_steps = 0
        if self.bs_workload or "bsNumSamples" in settings:
            self.bs_n_samples = _read_positive_int(
                settings=settings, key="bsNumSamples"
            )
        if self.bs_workload or "bsNumSteps" in settings:
            self.bs_n_steps = _read_positive_int(settings=settings, key="bsNumSteps")

    def compute_workload(self, market: Market) -> float:
        """Process the workload for the market.

        The pseudo random number generator of the Monte Carlo simulation is seeded with
        a string of the agent ID, the market ID and the time. Thus, the agent's own
        pseudo random number generator is left untouched, and the same work is done
        whichever runner is used. String seeds do not depend on PYTHONHASHSEED.

        Args:
            market (Market): market to order.

        Returns:
            float: estimated price of the option.

        """
        prng = random.Random(  # nosec B311 # a workload, not for security
            f"{self.agent_id}:{market.market_id}:{market.get_time()}"
        )
        return black_scholes_monte_carlo(
            prng=prng, n_samples=self.bs_n_samples, n_steps=self.bs_n_steps
        )

    def submit_orders_by_market(self, market: Market) -> List[Union[Order, Cancel]]:
        """Process the workload and submit orders with the probability ``order_rate``.

        The orders are made by :func:`pams.agents.FCNAgent.submit_orders_by_market`.
        If ``order_rate`` is 1.0, no random number is drawn for the decision, so the
        agent behaves exactly as FCN agents.

        Args:
            market (Market): market to order.

        Returns:
            List[Union[Order, Cancel]]: order list.

        """
        if not self.is_market_accessible(market_id=market.market_id):
            return []
        if self.bs_workload:
            self.compute_workload(market=market)
        if self.order_rate < 1.0 and self.prng.random() > self.order_rate:
            return []
        return super().submit_orders_by_market(market=market)
