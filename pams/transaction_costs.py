"""Transaction costs charged by markets at each execution."""
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Tuple

from .order import Order

if TYPE_CHECKING:  # pragma: no cover
    from .market import Market


class TransactionCost(ABC):
    """Transaction cost base class (ABC class).

    It defines the transaction costs that the buyer and the seller pay at each execution in a market.
    All transaction costs should inherit this class and implement :func:`compute_costs`.

    A transaction cost is set by the "transactionCost" setting of a market group in the config, e.g.,
    ``{"class": "ProportionalTransactionCost", "rate": 0.001}``. The runner creates one instance for each
    market of the group and sets it to ``market.transaction_cost``. Then, it calls :func:`setup` with the
    setting without "class", right after the setup of the market. Without the setting,
    ``market.transaction_cost`` is None and no costs are charged.

    Because an instance is bound to its market, :func:`compute_costs` only receives the execution: the price,
    the volume and the two orders. Everything else can be read through ``self.market``, e.g., the order books
    (``self.market.buy_order_book`` and ``self.market.sell_order_book``), the prices, the time
    (``self.market.get_time()``), the simulator (``self.market.simulator``), the agents
    (``self.market.simulator.id2agent``) and the other markets (``self.market.simulator.markets``).
    An instance can also keep its own state, e.g., the cumulative volume of each agent for tiered fees.

    The market calls :func:`compute_costs` once for each execution and records the costs in
    :class:`pams.logs.ExecutionLog` as ``buy_transaction_cost`` and ``sell_transaction_cost``. The simulator
    subtracts them from the cash of the buyer and the seller. No agent receives them. Instead, the market adds both
    costs to the transaction costs that it collects in the time step, which
    :func:`pams.Market.get_transaction_cost_revenues` returns for each time step and
    :func:`pams.Market.get_cumulative_transaction_cost_revenue` returns as the balance up to a time step. Negative
    costs (rebates) reduce them.

    When the market executes orders, which the runner asks it to do after each order or cancel while the market
    is running, it first matches the best buy and sell orders as long as their prices cross and determines one
    execution price for all the matched pairs. Then, it executes the pairs one by one in the order in which they
    were matched, and :func:`compute_costs` is called for each pair before the volume of the pair is subtracted
    from its orders. At that moment:

    - the two orders of the pair and the orders of the later pairs are still in the order books, and their
      volumes still include this pair and the later pairs (only the earlier pairs are subtracted), so the order
      books can still be crossed;
    - the earlier pairs of the same matching are already executed: their volumes are subtracted from their
      orders, the fully executed orders are removed from the order books, and the executed prices and volumes
      and the transaction cost revenue of the market include them;
    - the cash and the asset volumes of the agents do not include any pair of the matching yet, because the
      simulator updates them after all the pairs are executed.

    Creating and setting up transaction costs does not draw random numbers from the pseudo random number
    generators of the runner or of the markets, and the built-in transaction costs draw none. Therefore, for the
    same seed, the prices and the executions are exactly the same with and without transaction costs, as long as
    the agents do not look at their cash (the built-in agents do not). If a transaction cost needs random
    numbers, create its own ``random.Random`` in :func:`setup`.

    .. seealso::
        - :class:`pams.transaction_costs.ProportionalTransactionCost`
    """

    def __init__(self, market: "Market") -> None:
        """Transaction cost initialization. Usually be called from runner automatically.

        Args:
            market (:class:`pams.Market`): market that charges this transaction cost.

        Returns:
            None

        """
        self.market: "Market" = market

    def __repr__(self) -> str:
        """Return the string representation of the transaction cost."""
        return f"<{self.__class__.__module__}.{self.__class__.__name__} | market={self.market}>"

    def setup(  # type: ignore  # noqa: B027
        self, settings: Dict[str, Any], *args, **kwargs
    ) -> None:
        """Transaction cost setup. Usually be called from runner automatically.

        It is called right after the setup of the market. By default, it does nothing.

        Args:
            settings (Dict[str, Any]): transaction cost configuration, i.e., the "transactionCost" setting of the
                                       market without "class". Usually, automatically set from json config of
                                       simulator.
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        """

    @abstractmethod
    def compute_costs(
        self, price: float, volume: int, buy_order: Order, sell_order: Order
    ) -> Tuple[float, float]:
        """Compute the transaction costs of an execution.

        This is called by the market for each execution, before the executed volume is subtracted from the
        orders (see :class:`TransactionCost` for the state of the market at that moment). You must implement this.

        Args:
            price (float): executed price.
            volume (int): executed volume.
            buy_order (:class:`pams.order.Order`): buy order. Its volume still includes the executed volume.
            sell_order (:class:`pams.order.Order`): sell order. Its volume still includes the executed volume.

        Returns:
            Tuple[float, float]: transaction costs of the buyer and of the seller, which must be finite numbers.
            Negative costs are rebates, which are added to the cash.

        """


class ProportionalTransactionCost(TransactionCost):
    """Proportional transaction cost class.

    Both the buyer and the seller of each execution pay ``rate`` times the executed value (``price * volume``).
    The rate is set by "rate" in the settings, e.g., ``{"class": "ProportionalTransactionCost", "rate": 0.001}``
    for 0.1%.

    .. seealso::
        - :class:`pams.transaction_costs.TransactionCost`
    """

    def __init__(self, market: "Market") -> None:
        """Transaction cost initialization. Usually be called from runner automatically.

        Args:
            market (:class:`pams.Market`): market that charges this transaction cost.

        Returns:
            None

        """
        super().__init__(market=market)
        self.rate: float = 0.0

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        """Transaction cost setup. Usually be called from runner automatically.

        Args:
            settings (Dict[str, Any]): transaction cost configuration. Usually, automatically set from json config
                                       of simulator. This must include the parameter "rate", a number in
                                       [0.0, 1.0).
            *args: not used.
            **kwargs: not used.

        Returns:
            None

        """
        if "rate" not in settings:
            raise ValueError("rate is required for ProportionalTransactionCost")
        rate = settings["rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError("rate must be int or float")
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0.0, 1.0)")
        self.rate = float(rate)

    def compute_costs(
        self, price: float, volume: int, buy_order: Order, sell_order: Order
    ) -> Tuple[float, float]:
        """Compute the transaction costs of an execution.

        Args:
            price (float): executed price.
            volume (int): executed volume.
            buy_order (:class:`pams.order.Order`): buy order (not used).
            sell_order (:class:`pams.order.Order`): sell order (not used).

        Returns:
            Tuple[float, float]: transaction costs of the buyer and of the seller, both ``rate * price * volume``.

        """
        cost: float = self.rate * price * volume
        return cost, cost
