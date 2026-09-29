from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pams.agents import FCNAgent
from pams.market import Market
from pams.order import Cancel
from pams.order import Order
from pams.utils.json_random import JsonRandom
from samples.black_scholes.black_scholes import BlackScholes

# ORDER_RATE of WorkloadFCNAgent in plhamJ
DEFAULT_ORDER_RATE: float = 0.1

# the option priced by the workload of WorkloadFCNAgent in plhamJ
BS_INITIAL_PRICE: float = 100.0
BS_STRIKE_PRICE: float = 100.0
BS_RISK_FREE_RATE: float = 0.1
BS_VOLATILITY: float = 0.3
BS_MATURITY_TIME: float = 3.0


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

    This agent mimics the burden of Monte Carlo simulations. Every time it is asked to
    submit orders while the workload is on (see :func:`is_workload_on`), it prices an
    option by :class:`samples.black_scholes.black_scholes.BlackScholes` with its own
    pseudo random number generator. Then, it submits orders in the same way as FCN
    agents with the probability ``order_rate``.
    This class inherits from the :class:`pams.agents.FCNAgent` class.

    .. note::
        The estimated option prices are summed up to ``bs_sum``. When this agent is used
        with :class:`pams.runners.MultiProcessAgentParallelRunner`, ``bs_sum`` is
        updated only on the copies of the agent on the worker processes and stays 0.0
        on the main process. The simulation results do not depend on ``bs_sum``.
    """

    bs_workload: bool
    bs_load_shutdown: int
    bs_n_samples: int
    bs_n_steps: int
    order_rate: float
    bs_sum: float

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
                parameters "bsWorkload" (false if not specified), "bsLoadShutdown"
                (-1 if not specified), "bsNumSamples", "bsNumSteps", and "orderRate"
                (0.1 if not specified) in addition to the parameters of
                :class:`pams.agents.FCNAgent`. "bsNumSamples" and "bsNumSteps" are
                required if "bsWorkload" is true.
            accessible_markets_ids (List[int]): list of market IDs.
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        self.bs_workload = settings.get("bsWorkload", False)
        if not isinstance(self.bs_workload, bool):
            raise ValueError("bsWorkload have to be bool")
        self.bs_load_shutdown = settings.get("bsLoadShutdown", -1)
        if isinstance(self.bs_load_shutdown, bool) or not isinstance(
            self.bs_load_shutdown, int
        ):
            raise ValueError("bsLoadShutdown have to be int")
        self.bs_n_samples = 0
        self.bs_n_steps = 0
        if self.bs_workload or "bsNumSamples" in settings:
            self.bs_n_samples = _read_positive_int(
                settings=settings, key="bsNumSamples"
            )
        if self.bs_workload or "bsNumSteps" in settings:
            self.bs_n_steps = _read_positive_int(settings=settings, key="bsNumSteps")
        if "orderRate" in settings:
            json_random: JsonRandom = JsonRandom(prng=self.prng)
            self.order_rate = json_random.random(json_value=settings["orderRate"])
        else:
            self.order_rate = DEFAULT_ORDER_RATE
        if not 0.0 <= self.order_rate <= 1.0:
            raise ValueError("orderRate have to be between 0.0 and 1.0")
        self.bs_sum = 0.0

    def is_workload_on(self, time: int) -> bool:
        """Determine if the workload is processed at the time.

        The workload is processed if ``bs_workload`` is true and the time is before
        ``bs_load_shutdown``. If ``bs_load_shutdown`` is 0 or negative, the workload is
        never shut down.

        Args:
            time (int): time.

        Returns:
            bool: whether the workload is processed or not.

        """
        return self.bs_workload and (
            self.bs_load_shutdown <= 0 or time < self.bs_load_shutdown
        )

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """Process the workload and submit orders with the probability ``order_rate``.

        If none of the markets is accessible, neither the workload is processed nor
        orders are submitted. The orders are made by
        :func:`pams.agents.FCNAgent.submit_orders`.

        Args:
            markets (List[Market]): markets to order.

        Returns:
            List[Union[Order, Cancel]]: order list.

        """
        accessible_markets: List[Market] = [
            market
            for market in markets
            if self.is_market_accessible(market_id=market.market_id)
        ]
        if len(accessible_markets) == 0:
            return []
        if self.is_workload_on(time=accessible_markets[0].get_time()):
            black_scholes = BlackScholes(
                prng=self.prng,
                initial_price=BS_INITIAL_PRICE,
                strike_price=BS_STRIKE_PRICE,
                risk_free_rate=BS_RISK_FREE_RATE,
                volatility=BS_VOLATILITY,
                maturity_time=BS_MATURITY_TIME,
            )
            self.bs_sum += black_scholes.compute(
                n_samples=self.bs_n_samples, n_steps=self.bs_n_steps
            )
        # randomly submit (or not) orders
        if self.prng.random() > self.order_rate:
            return []
        return super().submit_orders(markets=markets)
