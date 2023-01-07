from queue import PriorityQueue
from typing import Dict
from typing import List
from typing import Optional

from .order import Cancel
from .order import Order


class OrderBook:
    def __init__(self, is_buy: bool) -> None:
        self.priority_queue: PriorityQueue[Order] = PriorityQueue()
        self.time: int = 0
        self.is_buy = is_buy
        self.expire_time_list: Dict[int, List[Order]] = {}

    def add(self, order: Order) -> None:
        if order.is_buy != self.is_buy:
            raise ValueError("buy/sell is incorrect")
        order.placed_at = self.time
        self.priority_queue.put(order)
        if order.ttl is not None:
            expiration_time = order.placed_at + order.ttl
            if expiration_time not in self.expire_time_list:
                self.expire_time_list[expiration_time] = []
            self.expire_time_list[expiration_time].append(order)

    def _remove(self, order: Order) -> None:
        self.priority_queue.queue.remove(order)
        if order.placed_at is None:
            raise AssertionError("the order is not yet placed")
        if order.ttl is not None:
            expiration_time = order.placed_at + order.ttl
            self.expire_time_list[expiration_time].remove(order)

    def cancel(self, cancel: Cancel) -> None:
        cancel.order.is_canceled = True
        cancel.placed_at = self.time
        self._remove(cancel.order)

    def get_best_order(self) -> Optional[Order]:
        if len(self.priority_queue.queue) > 0:
            return self.priority_queue.queue[0]
        else:
            return None

    def get_best_price(self) -> Optional[float]:
        if len(self.priority_queue.queue) > 0:
            return self.priority_queue.queue[0].price
        else:
            return None

    def change_order_volume(self, order: Order, delta: int) -> None:
        order.volume += delta
        if order.volume == 0:
            self._remove(order=order)

    def _check_expired_orders(self) -> None:
        delete_orders: List[Order] = sum(
            [value for key, value in self.expire_time_list.items() if key < self.time],
            [],
        )
        delete_keys: List[int] = [
            key for key, value in self.expire_time_list.items() if key < self.time
        ]
        for delete_order in delete_orders:
            self.priority_queue.queue.remove(delete_order)
        for key in delete_keys:
            self.expire_time_list.pop(key)

    def _set_time(self, time: int) -> None:
        self.time = time
        self._check_expired_orders()

    def _update_time(self) -> None:
        self.time += 1
        self._check_expired_orders()

    def __len__(self) -> int:
        return len(self.priority_queue.queue)

    def get_price_volume(self) -> Dict[Optional[float], int]:
        keys: List[Optional[float]] = list(
            set(map(lambda x: x.price, self.priority_queue.queue))
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
                            for order in self.priority_queue.queue
                            if order.price == key
                        ]
                    ),
                )
                for key in keys
            ]
        )
        return result
