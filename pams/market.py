import heapq
import math
import random
import warnings
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import cast

from .logs.base import CancelLog
from .logs.base import ExecutionLog
from .logs.base import ExpirationLog
from .logs.base import Log
from .logs.base import Logger
from .logs.base import OrderLog
from .order import Cancel
from .order import Order
from .order_book import OrderBook

T = TypeVar("T")


class Market:
    """Market class.

    .. seealso::
        - :class:`pams.index_market.IndexMarket`: IndexMarket
    """

    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """initialization.

        Args:
            market_id (int): market ID.
            prng (random.Random): pseudo random number generator for this market.
            simulator (:class:`pams.Simulator`): simulator that executes this market.
            name (str): market name.
            logger (Logger, Optional): logger.

        Returns:
            None
        """
        self.market_id: int = market_id
        self._prng = prng
        self.logger: Optional[Logger] = logger
        self._is_running: bool = False
        self.tick_size: float = 1.0
        self.chunk_size = 100
        self.sell_order_book: OrderBook = OrderBook(is_buy=False)
        self.buy_order_book: OrderBook = OrderBook(is_buy=True)
        self.time: int = -1
        self._market_prices: List[Optional[float]] = []
        self._last_executed_prices: List[Optional[float]] = []
        self._mid_prices: List[Optional[float]] = []
        self._fundamental_prices: List[Optional[float]] = []
        self._executed_volumes: List[int] = []
        self._executed_total_prices: List[float] = []
        self._n_buy_orders: List[int] = []
        self._n_sell_orders: List[int] = []
        self._next_order_id: int = 0
        self.simulator: "Simulator" = simulator  # type: ignore  # NOQA
        self.name: str = name
        self.outstanding_shares: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.market_id}, name={self.name}, "
            f"tick_size={self.tick_size}, outstanding_shares={self.outstanding_shares}>"
        )

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """setup market configuration from setting format.

        Args:
            settings (Dict[str, Any]): market configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "tickSize" and either "marketPrice" or "fundamentalPrice".
                                       This can include the parameter "outstandingShares" and "tradeVolume".

        Returns:
            None
        """
        if "tickSize" not in settings:
            raise ValueError("tickSize is required")
        self.tick_size = settings["tickSize"]
        if "outstandingShares" in settings:
            if not isinstance(settings["outstandingShares"], int):
                raise ValueError("outstandingShares must be int")
            self.outstanding_shares = settings["outstandingShares"]
        if "marketPrice" in settings:
            self._market_prices = [float(settings["marketPrice"])]
        elif "fundamentalPrice" in settings:
            self._market_prices = [float(settings["fundamentalPrice"])]
        else:
            raise ValueError("fundamentalPrice or marketPrice is required for market")

    def _extract_sequential_data_by_time(
        self,
        times: Union[Iterable[int], None],
        parameters: List[Optional[T]],
        allow_none: bool = False,
    ) -> List[Optional[T]]:
        """extract sequential parameters by time. (Internal method)

        Args:
            times (Union[Iterable[int], None]): range of time steps.
            parameters (List[Optional[T]]): referenced parameters.
            allow_none (bool): whether a None result can be returned.

        Returns:
            List[Optional[T]]: extracted parameters.
        """
        if times is None:
            times = range(self.time + 1)
        if sum([t > self.time for t in times]) > 0:
            raise AssertionError("Cannot refer the future parameters")
        result = [parameters[t] for t in times]
        if not allow_none and None in result:
            raise AssertionError
        return result

    def _extract_data_by_time(
        self,
        time: Union[int, None],
        parameters: List[Optional[T]],
        allow_none: bool = False,
    ) -> Optional[T]:
        """extract a parameter by time. (Internal method)

        Args:
            time (Union[int, None]): time step.
            parameters (List[Optional[T]]): referenced parameters.
            allow_none (bool): whether a None result can be returned.

        Returns:
            Optional[T]: extracted parameter.
        """
        if time is None:
            time = self.time
        if time > self.time:
            raise AssertionError("Cannot refer the future parameters")
        result = parameters[time]
        if not allow_none and result is None:
            raise AssertionError
        return result

    def get_time(self) -> int:
        """get time step."""
        return self.time

    def get_market_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[float]:
        """get market prices.

        Args:
            times (Union[Iterable[int], None]): range of time steps.

        Returns:
            List[float]: extracted sequential data.
        """
        return cast(
            List[float],
            self._extract_sequential_data_by_time(
                times, self._market_prices, allow_none=False
            ),
        )

    def get_market_price(self, time: Union[int, None] = None) -> float:
        """get market price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float: extracted data.
        """
        return cast(
            float,
            self._extract_data_by_time(time, self._market_prices, allow_none=False),
        )

    def get_mid_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[Optional[float]]:
        """get middle prices.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[Optional[float]]: middle prices.
        """
        return self._extract_sequential_data_by_time(
            times, self._mid_prices, allow_none=True
        )

    def get_mid_price(self, time: Union[int, None] = None) -> Optional[float]:
        """get middle price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float, Optional: middle price.
        """
        return self._extract_data_by_time(time, self._mid_prices, allow_none=True)

    def get_last_executed_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[Optional[float]]:
        """get prices executed last steps.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[Optional[float]]: prices.
        """
        return self._extract_sequential_data_by_time(
            times, self._last_executed_prices, allow_none=True
        )

    def get_last_executed_price(self, time: Union[int, None] = None) -> Optional[float]:
        """get price executed last step.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float, Optional: price.
        """
        return self._extract_data_by_time(
            time, self._last_executed_prices, allow_none=True
        )

    def get_fundamental_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[float]:
        """get fundamental prices.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[float]: fundamental prices.
        """
        return cast(
            List[float],
            self._extract_sequential_data_by_time(times, self._fundamental_prices),
        )

    def get_fundamental_price(self, time: Union[int, None] = None) -> float:
        """get fundamental price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float: fundamental price.
        """
        return cast(float, self._extract_data_by_time(time, self._fundamental_prices))

    def get_executed_volumes(
        self, times: Union[Iterable[int], None] = None
    ) -> List[int]:
        """get executed volumes.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[int]: volumes.
        """
        return cast(
            List[int],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._executed_volumes)
            ),
        )

    def get_executed_volume(self, time: Union[int, None] = None) -> int:
        """get executed volume.

        Args:
            time (Union[int, None]): time step.

        Returns:
            int: volume.
        """
        return cast(
            int,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._executed_volumes)
            ),
        )

    def get_executed_total_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[float]:
        """get executed total prices.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[float]: total prices.
        """
        return cast(
            List[float],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._executed_total_prices)
            ),
        )

    def get_executed_total_price(self, time: Union[int, None] = None) -> float:
        """get executed total price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float: total price.
        """
        return cast(
            float,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._executed_total_prices)
            ),
        )

    def get_n_buy_orders(self, times: Union[Iterable[int], None] = None) -> List[int]:
        """get the number of buy orders.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[int]: number of buy orders.
        """
        return cast(
            List[int],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._n_buy_orders)
            ),
        )

    def get_n_buy_order(self, time: Union[int, None] = None) -> int:
        """get the number of buy order.

        Args:
            time (Union[int, None]): time step.

        Returns:
            int: number of buy order.
        """
        return cast(
            int,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._n_buy_orders)
            ),
        )

    def get_n_sell_orders(self, times: Union[Iterable[int], None] = None) -> List[int]:
        """get the number of sell orders.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[int]: number of sell orders.
        """
        return cast(
            List[int],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._n_sell_orders)
            ),
        )

    def get_n_sell_order(self, time: Union[int, None] = None) -> int:
        """get the number of sell order.

        Args:
            time (Union[int, None]): time step.

        Returns:
            int: number of sell order.
        """
        return cast(
            int,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._n_sell_orders)
            ),
        )

    def _fill_until(self, time: int) -> None:
        if len(self._mid_prices) >= time + 1:
            return
        length = (time // self.chunk_size + 1) * self.chunk_size
        self._market_prices = self._market_prices + [
            None for _ in range(length - len(self._market_prices))
        ]
        self._mid_prices = self._mid_prices + [
            None for _ in range(length - len(self._mid_prices))
        ]
        self._last_executed_prices = self._last_executed_prices + [
            None for _ in range(length - len(self._last_executed_prices))
        ]
        self._fundamental_prices = self._fundamental_prices + [
            None for _ in range(length - len(self._fundamental_prices))
        ]
        self._executed_volumes = self._executed_volumes + [
            0 for _ in range(length - len(self._executed_volumes))
        ]
        self._executed_total_prices = self._executed_total_prices + [
            0 for _ in range(length - len(self._executed_total_prices))
        ]
        self._n_buy_orders = self._n_buy_orders + [
            0 for _ in range(length - len(self._n_buy_orders))
        ]
        self._n_sell_orders = self._n_sell_orders + [
            0 for _ in range(length - len(self._n_sell_orders))
        ]

    def get_vwap(self, time: Optional[int] = None) -> float:
        """get VWAP.

        Args:
            time (int, Optional): time step.

        Returns:
            float: VWAP.
        """
        if time is None:
            time = self.time
        if time > self.time:
            raise AssertionError("Cannot refer the future parameters")
        if sum(self._executed_volumes[: time + 1]) == 0:
            return float("nan")
        return sum(self._executed_total_prices[: time + 1]) / sum(
            self._executed_volumes[: time + 1]
        )

    @property
    def is_running(self) -> bool:
        """get whether this market is running or not.

        Returns:
            bool: whether this market is running or not.
        """
        return self._is_running

    def get_best_buy_price(self) -> Optional[float]:
        """get the best buy price.

        Returns:
            float, Optional: the best buy price.
        """
        return self.buy_order_book.get_best_price()

    def get_best_sell_price(self) -> Optional[float]:
        """get the best sell price.

        Returns:
            float, Optional: the best sell price.
        """
        return self.sell_order_book.get_best_price()

    def get_sell_order_book(self) -> Dict[Optional[float], int]:
        """get sell order book.

        Returns:
            Dict[Optional[float], int]: sell order book.
        """
        return self.sell_order_book.get_price_volume()

    def get_buy_order_book(self) -> Dict[Optional[float], int]:
        """get buy order book.

        Returns:
            Dict[Optional[float], int]: buy order book.
        """
        return self.buy_order_book.get_price_volume()

    def convert_to_tick_level_rounded_lower(self, price: float) -> int:
        """convert price to tick level rounded lower.

        Args:
            price (float): price.

        Returns:
            int: price for tick level rounded lower.
        """
        return math.floor(price / self.tick_size)

    def convert_to_tick_level_rounded_upper(self, price: float) -> int:
        """convert price to tick level rounded upper.

        Args:
            price (float): price.

        Returns:
            int: price for tick level rounded upper.
        """
        return math.ceil(price / self.tick_size)

    def convert_to_tick_level(self, price: float, is_buy: bool) -> int:
        """convert price to tick level. If it is buy order price, it is rounded lower. If it is sell order price, it is rounded upper.

        Args:
            price (float): price.
            is_buy (bool): buy order or not.

        Returns:
            int: price for tick level.
        """
        if is_buy:
            return self.convert_to_tick_level_rounded_lower(price=price)
        else:
            return self.convert_to_tick_level_rounded_upper(price=price)

    def convert_to_price(self, tick_level: int) -> float:
        """convert tick to price.

        Args:
            tick_level (int): tick level.

        Returns:
            float: price.
        """
        return self.tick_size * tick_level

    def _set_time(self, time: int, next_fundamental_price: float) -> None:
        """set time step. (Usually, only triggered by simulator)

        Args:
            time (int): time step.
            next_fundamental_price (float): next fundamental price.

        Returns:
            None
        """
        self.time = time
        logs: List[ExpirationLog] = self.buy_order_book._set_time(time)
        if self.logger is not None:
            for log in logs:
                log.read_and_write(logger=self.logger)
        logs_: List[ExpirationLog] = self.sell_order_book._set_time(time)
        if self.logger is not None:
            for log_ in logs_:
                log_.read_and_write(logger=self.logger)
        self._fill_until(time=time)
        self._fundamental_prices[self.time] = next_fundamental_price
        if self.time > 0:
            executed_prices: List[float] = cast(
                List[float],
                list(
                    filter(
                        lambda x: x is not None, self._last_executed_prices[: self.time]
                    )
                ),
            )
            self._last_executed_prices[self.time] = (
                executed_prices[-1] if sum(executed_prices) > 0 else None
            )
            mid_prices: List[float] = cast(
                List[float],
                list(filter(lambda x: x is not None, self._mid_prices[: self.time])),
            )
            self._mid_prices[self.time] = (
                mid_prices[-1] if sum(mid_prices) > 0 else None
            )
            market_prices: List[float] = cast(
                List[float],
                list(filter(lambda x: x is not None, self._market_prices[: self.time])),
            )
            self._market_prices[self.time] = (
                market_prices[-1] if sum(market_prices) > 0 else None
            )
            if self.is_running:
                if self._last_executed_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._last_executed_prices[
                        self.time
                    ]
                elif self._mid_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._mid_prices[self.time]

    def _update_time(self, next_fundamental_price: float) -> None:
        """update time. (Usually, only triggered by simulator)

        Args:
            next_fundamental_price (float): next fundamental price.

        Returns:
            None
        """
        self.time += 1
        logs: List[ExpirationLog] = self.buy_order_book._set_time(self.time)
        if self.logger is not None:
            for log in logs:
                log.read_and_write(logger=self.logger)
        logs_: List[ExpirationLog] = self.sell_order_book._set_time(self.time)
        if self.logger is not None:
            for log_ in logs_:
                log_.read_and_write(logger=self.logger)
        self._fill_until(time=self.time)
        self._fundamental_prices[self.time] = next_fundamental_price
        if self.time > 0:
            self._last_executed_prices[self.time] = self._last_executed_prices[
                self.time - 1
            ]
            self._mid_prices[self.time] = self._mid_prices[self.time - 1]
            self._market_prices[self.time] = self._market_prices[self.time - 1]
            if self.is_running:
                if self._last_executed_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._last_executed_prices[
                        self.time - 1
                    ]
                elif self._mid_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._mid_prices[self.time - 1]
        else:
            if self._market_prices[self.time] is None:
                self._market_prices[self.time] = next_fundamental_price

    def _cancel_order(self, cancel: Cancel) -> CancelLog:
        """cancel order. (Usually, only triggered by simulator)

        Args:
            cancel (:class:`pams.order.Cancel`): cancel class.

        Returns:
            :class:`pams.logs.base.CancelLog`: cancel log.
        """
        if self.market_id != cancel.order.market_id:
            raise ValueError("this cancel order is for a different market")
        if cancel.order.order_id is None or cancel.order.placed_at is None:
            raise ValueError("the order is not submitted before")
        (self.buy_order_book if cancel.order.is_buy else self.sell_order_book).cancel(
            cancel=cancel
        )
        if cancel.placed_at is None:
            raise AssertionError
        self._update_market_price()

        log: CancelLog = CancelLog(
            order_id=cancel.order.order_id,
            market_id=cancel.order.market_id,
            cancel_time=cancel.placed_at,
            order_time=cancel.order.placed_at,
            agent_id=cancel.order.agent_id,
            is_buy=cancel.order.is_buy,
            kind=cancel.order.kind,
            volume=cancel.order.volume,
            price=cancel.order.price,
            ttl=cancel.order.ttl,
        )
        if self.logger is not None:
            log.read_and_write(logger=self.logger)
        return log

    def _update_market_price(self) -> None:
        """update market price. (Internal method)"""
        best_buy_price: Optional[float] = self.get_best_buy_price()
        best_sell_price: Optional[float] = self.get_best_sell_price()
        if best_buy_price is None or best_sell_price is None:
            self._mid_prices[self.time] = None
        else:
            self._mid_prices[self.time] = (
                (best_sell_price + best_buy_price) / 2.0
                if best_sell_price is not None and best_buy_price is not None
                else None
            )
        if self.is_running:
            if self._last_executed_prices[self.time] is not None:
                self._market_prices[self.time] = self._last_executed_prices[self.time]
            elif self._mid_prices[self.time] is not None:
                self._market_prices[self.time] = self._mid_prices[self.time]

    def _execute_orders(
        self, price: float, volume: int, buy_order: Order, sell_order: Order
    ) -> ExecutionLog:
        """execute orders. (Internal method)

        Args:
            price (float): price.
            volume (int): volume.
            buy_order (:class:`pams.order.Order`): buy order.
            sell_order (:class:`pams.order.Order`): sell order.

        Returns:
            :class:`pams.logs.base.CancelLog`: execution log.
        """
        if not self.is_running:
            raise AssertionError("market is not running")
        if buy_order.market_id != self.market_id:
            raise ValueError("buy order is not for this market")
        if sell_order.market_id != self.market_id:
            raise ValueError("sell order is not for this market")

        if buy_order.placed_at is None:
            raise ValueError("buy order is not submitted yet")
        if sell_order.placed_at is None:
            raise ValueError("sell order is not submitted yet")

        if volume <= 0:
            raise AssertionError

        log: ExecutionLog = ExecutionLog(
            market_id=self.market_id,
            time=self.time,
            buy_agent_id=buy_order.agent_id,
            sell_agent_id=sell_order.agent_id,
            buy_order_id=cast(int, buy_order.order_id),
            sell_order_id=cast(int, sell_order.order_id),
            price=price,
            volume=volume,
        )

        self.buy_order_book.change_order_volume(order=buy_order, delta=-volume)
        self.sell_order_book.change_order_volume(order=sell_order, delta=-volume)

        self._last_executed_prices[self.time] = price
        self._executed_volumes[self.time] += volume
        self._executed_total_prices[self.time] += volume * price
        self._update_market_price()

        # ToDo: Agent modification will be handled in simulator
        if self.logger is not None:
            log.read_and_write(logger=self.logger)
        return log

    def _add_order(self, order: Order) -> OrderLog:
        """add order. (Usually, only triggered by runner)

        Args:
            order (:class:`pams.order.Order`): order.

        Returns:
            :class:`pams.logs.base.OrderLog`: order log.
        """
        if order.market_id != self.market_id:
            raise ValueError("order is not for this market")
        if order.placed_at is not None:
            raise ValueError("the order is already submitted")
        if order.order_id is not None:
            raise ValueError("the order is already submitted")
        if order.price is not None and order.price % self.tick_size != 0:
            warnings.warn(
                "order price does not accord to the tick size. price will be modified"
            )
            order.price = (
                self.convert_to_tick_level(price=order.price, is_buy=order.is_buy)
                * self.tick_size
            )
        order.order_id = self._next_order_id
        self._next_order_id += 1
        (self.buy_order_book if order.is_buy else self.sell_order_book).add(order=order)
        if order.placed_at != self.time:
            raise AssertionError
        self._update_market_price()
        if order.is_buy:
            self._n_buy_orders[self.time] += 1
        else:
            self._n_sell_orders[self.time] += 1

        log: OrderLog = OrderLog(
            order_id=order.order_id,
            market_id=order.market_id,
            time=cast(int, order.placed_at),
            agent_id=order.agent_id,
            is_buy=order.is_buy,
            kind=order.kind,
            volume=order.volume,
            price=order.price,
            ttl=order.ttl,
        )
        if self.logger is not None:
            log.read_and_write(logger=self.logger)
        return log

    def remain_executable_orders(self) -> bool:
        """check if there are remain executable orders in this market.

        Returns:
            bool: whether some orders is executable or not.
        """
        if len(self.sell_order_book) == 0:
            return False
        if len(self.buy_order_book) == 0:
            return False
        sell_best: Order = cast(Order, self.sell_order_book.get_best_order())
        buy_best: Order = cast(Order, self.buy_order_book.get_best_order())
        if sell_best.price is not None or buy_best.price is not None:
            if sell_best.price is not None and buy_best.price is not None:
                return sell_best.price <= buy_best.price
            else:
                return True
        else:
            sell_book: Dict[
                Optional[float], int
            ] = self.sell_order_book.get_price_volume()
            buy_book: Dict[
                Optional[float], int
            ] = self.buy_order_book.get_price_volume()
            if None not in sell_book or None not in buy_book:
                raise AssertionError
            if sell_book[None] != buy_book[None]:
                if sell_book[None] < buy_book[None]:
                    additional_required_orders = buy_book[None] - sell_book[None]
                    sell_book.pop(None)
                    return len(sell_book) >= additional_required_orders
                else:
                    additional_required_orders = sell_book[None] - buy_book[None]
                    buy_book.pop(None)
                    return len(buy_book) >= additional_required_orders
            else:
                sell_book.pop(None)
                buy_book.pop(None)
                if len(sell_book) == 0 or len(buy_book) == 0:
                    return False
                return min(list(cast(Dict[float, int], sell_book).keys())) <= max(
                    list(cast(Dict[float, int], buy_book).keys())
                )

    def _execution(self) -> List[ExecutionLog]:
        """execute for market. (Usually, only triggered by runner)

        Returns:
            List[:class:`pams.logs.base.ExecutionLog`]: execution logs.
        """
        if not self.remain_executable_orders():
            return []
        pending: List[Tuple[int, Order, Order]] = []

        popped_buy_orders: List[Order] = []
        popped_sell_orders: List[Order] = []

        buy_order: Order = heapq.heappop(self.buy_order_book.priority_queue)
        popped_buy_orders.append(buy_order)
        sell_order: Order
        buy_order_volume_tmp: int = buy_order.volume
        sell_order_volume_tmp: int = 0
        price: Optional[float] = None
        while True:
            if buy_order_volume_tmp != 0 and sell_order_volume_tmp != 0:
                raise AssertionError
            if buy_order_volume_tmp == 0:
                if len(self.buy_order_book.priority_queue) == 0:
                    break
                buy_order = heapq.heappop(self.buy_order_book.priority_queue)
                popped_buy_orders.append(buy_order)
                buy_order_volume_tmp = buy_order.volume
                if buy_order_volume_tmp == 0:
                    raise AssertionError
            if sell_order_volume_tmp == 0:
                if len(self.sell_order_book.priority_queue) == 0:
                    break
                sell_order = heapq.heappop(self.sell_order_book.priority_queue)
                popped_sell_orders.append(sell_order)
                sell_order_volume_tmp = sell_order.volume
                if sell_order_volume_tmp == 0:
                    raise AssertionError
            if (
                buy_order.price is not None
                and sell_order.price is not None
                and buy_order.price < sell_order.price
            ):
                break
            volume = min(buy_order_volume_tmp, sell_order_volume_tmp)
            if volume == 0:
                raise AssertionError
            buy_order_volume_tmp -= volume
            sell_order_volume_tmp -= volume
            if buy_order_volume_tmp < 0:
                raise AssertionError
            if sell_order_volume_tmp < 0:
                raise AssertionError
            if buy_order.price is None or sell_order.price is None:
                if buy_order.price is None and sell_order.price is None:
                    pending.append((volume, buy_order, sell_order))
                else:
                    price = (
                        buy_order.price
                        if buy_order.price is not None
                        else sell_order.price
                    )
                    pending.append((volume, buy_order, sell_order))
            else:
                if buy_order.placed_at == sell_order.placed_at:
                    if buy_order.order_id is None or sell_order.order_id is None:
                        raise AssertionError
                    if buy_order.order_id < sell_order.order_id:
                        price = buy_order.price
                    elif buy_order.order_id > sell_order.order_id:
                        price = sell_order.price
                    else:
                        raise AssertionError
                else:
                    price = (
                        buy_order.price
                        if cast(int, buy_order.placed_at)
                        < cast(int, sell_order.placed_at)
                        else sell_order.price
                    )
                pending.append((volume, buy_order, sell_order))
        if price is None:
            raise AssertionError
        # TODO: faster impl
        self.buy_order_book.priority_queue = [
            *popped_buy_orders,
            *self.buy_order_book.priority_queue,
        ]
        self.sell_order_book.priority_queue = [
            *popped_sell_orders,
            *self.sell_order_book.priority_queue,
        ]
        heapq.heapify(self.buy_order_book.priority_queue)
        heapq.heapify(self.sell_order_book.priority_queue)
        logs: List[ExecutionLog] = list(
            map(
                lambda x: self._execute_orders(
                    price=cast(float, price),
                    volume=x[0],
                    buy_order=x[1],
                    sell_order=x[2],
                ),
                pending,
            )
        )
        if self.remain_executable_orders():
            raise AssertionError
        if self.logger is not None:
            self.logger.bulk_write(logs=cast(List[Log], logs))
        return logs

    def change_fundamental_price(self, scale: float) -> None:
        """change fundamental price.

        Args:
            scale (float): scale.

        Returns:
            None
        """
        time: int = self.time
        current_fundamental: float = self.get_fundamental_price(time=time)
        new_fundamental: float = current_fundamental * scale
        self._fundamental_prices[time] = new_fundamental
        self.simulator.fundamentals.prices[self.market_id][time] = new_fundamental
        self.simulator.fundamentals._generated_until = time
