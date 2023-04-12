import heapq
from typing import Dict
from typing import List
from typing import Optional

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

    def _check_expired_orders(self) -> None:
        """check and delete expired orders. (Internal Method)"""
        delete_orders: List[Order] = sum(
            [value for key, value in self.expire_time_list.items() if key < self.time],
            [],
        )
        delete_keys: List[int] = [
            key for key, value in self.expire_time_list.items() if key < self.time
        ]
        if len(delete_orders) == 0:
            return
        # TODO: think better sorting in the following 3 lines
        includes_peek = self.priority_queue[0] in delete_orders
        for delete_order in delete_orders:
            self.priority_queue.remove(delete_order)
        if includes_peek:
            heapq.heapify(self.priority_queue)
        for key in delete_keys:
            self.expire_time_list.pop(key)

    def _set_time(self, time: int) -> None:
        """set time step. (Usually, it is called from market.)

        Args:
            time (int): time step.

        Returns:
            None
        """
        self.time = time
        self._check_expired_orders()

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
