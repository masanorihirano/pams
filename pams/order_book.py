import heapq
from typing import Dict
from typing import List
from typing import Optional

from .logs.base import ExpirationLog
from .order import Cancel
from .order import Order


class OrderBook:
    """Order book class."""

    def __init__(self, is_buy: bool) -> None:
        """initialization.

        Args:
            is_buy (bool): whether it is a buy order or not.

        Returns:
            None
        """
        self.priority_queue: List[Order] = []
        heapq.heapify(self.priority_queue)
        self.time: int = 0
        self.is_buy = is_buy
        self.expire_time_list: Dict[int, List[Order]] = {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__module__}.{self.__class__.__name__} | is_buy={self.is_buy}>"

    def add(self, order: Order) -> None:
        """add the book of order.

        Args:
            order (:class:`pams.order.Order`): order.

        Returns:
            None
        """
        if order.is_buy != self.is_buy:
            raise ValueError("buy/sell is incorrect")
        order.placed_at = self.time
        heapq.heappush(self.priority_queue, order)
        if order.ttl is not None:
            expiration_time = order.placed_at + order.ttl
            if expiration_time not in self.expire_time_list:
                self.expire_time_list[expiration_time] = []
            self.expire_time_list[expiration_time].append(order)

    def _remove(self, order: Order) -> None:
        """remove the book of order. (Internal method. Usually, it is not called from the outside of this class.)

        Args:
            order (:class:`pams.order.Order`): order.

        Returns:
            None
        """
        if order == self.priority_queue[0]:
            x = heapq.heappop(self.priority_queue)
            assert x == order
        else:
            self.priority_queue.remove(order)
            heapq.heapify(self.priority_queue)
        if order.placed_at is None:
            raise AssertionError("the order is not yet placed")
        if order.ttl is not None:
            expiration_time = order.placed_at + order.ttl
            self.expire_time_list[expiration_time].remove(order)

    def cancel(self, cancel: Cancel) -> None:
        """cancel the book of order.

        Args:
            cancel (:class:`pams.order.Cancel`): cancel order.

        Returns:
            None
        """
        cancel.order.is_canceled = True
        cancel.placed_at = self.time
        if cancel.order in self.priority_queue:
            # in case that order is executed before canceling.
            self._remove(cancel.order)

    def get_best_order(self) -> Optional[Order]:
        """get the order with the highest priority.

        Returns:
            :class:`pams.order.Order`, Optional: the order with the highest priority.
        """
        if len(self.priority_queue) > 0:
            return self.priority_queue[0]
        else:
            return None

    def get_best_price(self) -> Optional[float]:
        """get the order price with the highest priority.

        Returns:
            float, Optional: the order price with the highest priority.
        """
        if len(self.priority_queue) > 0:
            return self.priority_queue[0].price
        else:
            return None

    def change_order_volume(self, order: Order, delta: int) -> None:
        """change order volume.

        Args:
            order (:class:`pams.order.Order`): order.
            delta (int): amount of volume change.

        Returns:
            None
        """
        order.volume += delta
        # ToDo: check if volume is non-negative
        if order.volume == 0:
            self._remove(order=order)
        if order.volume < 0:
            raise AssertionError

    def _check_expired_orders(self) -> List[ExpirationLog]:
        """check and delete expired orders. (Internal Method)

        Returns:
            List[ExpirationLog]: the list of expiration logs.
        """
        delete_orders: List[Order] = sum(
            [value for key, value in self.expire_time_list.items() if key < self.time],
            [],
        )
        delete_keys: List[int] = [
            key for key, value in self.expire_time_list.items() if key < self.time
        ]
        logs: List[ExpirationLog] = []
        if len(delete_orders) == 0:
            return logs
        # TODO: think better sorting in the following 3 lines
        for delete_order in delete_orders:
            log: ExpirationLog = ExpirationLog(
                order_id=delete_order.order_id,
                market_id=delete_order.market_id,
                time=self.time,
                order_time=delete_order.placed_at,
                agent_id=delete_order.agent_id,
                is_buy=delete_order.is_buy,
                kind=delete_order.kind,
                volume=delete_order.volume,
                price=delete_order.price,
                ttl=delete_order.ttl,
            )
            logs.append(log)
            self.priority_queue.remove(delete_order)
        heapq.heapify(self.priority_queue)
        for key in delete_keys:
            self.expire_time_list.pop(key)
        return logs

    def _set_time(self, time: int) -> List[ExpirationLog]:
        """set time step. (Usually, it is called from market.)

        Args:
            time (int): time step.

        Returns:
            List[ExpirationLog]: the list of expiration logs.
        """
        self.time = time
        logs: List[ExpirationLog] = self._check_expired_orders()
        return logs

    def _update_time(self) -> None:
        """update time. (Usually, it is called from market.)
        Advance the time step and check expired orders.
        """
        self.time += 1
        self._check_expired_orders()

    def __len__(self) -> int:
        """get length of the order queue.

        Returns:
            int: length of the order queue.
        """
        return len(self.priority_queue)

    def get_price_volume(self) -> Dict[Optional[float], int]:
        """get price and volume (order book).

        Returns:
            Dict[Optional[float], int]: order book dict. Dict key is order price and the value is volumes.
        """
        keys: List[Optional[float]] = list(
            set(map(lambda x: x.price, self.priority_queue))
        )
        has_market_order: bool = None in keys
        if has_market_order:
            keys.remove(None)
        keys.sort(reverse=self.is_buy)
        if has_market_order:
            keys.insert(0, None)
        result: Dict[Optional[float], int] = dict(
            [
                (
                    key,
                    sum(
                        [
                            order.volume
                            for order in self.priority_queue
                            if order.price == key
                        ]
                    ),
                )
                for key in keys
            ]
        )
        return result
