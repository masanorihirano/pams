from queue import PriorityQueue
from typing import Optional

from .order import Cancel
from .order import Order


class OrderBook:
    def __init__(self, is_buy: bool) -> None:
        self.priority_queue: PriorityQueue[Order] = PriorityQueue()
        self.time: int = 0
        self.is_buy = is_buy

    def add(self, order: Order) -> None:
        if order.is_buy != self.is_buy:
            raise ValueError("buy/sell is incorrect")
        order.placed_at = self.time
        self.priority_queue.put(order)

    def remove(self, order: Order) -> None:
        self.priority_queue.queue.remove(order)

    def cancel(self, cancel: Cancel) -> None:
        cancel.order.is_canceled = True
        self.remove(cancel.order)

    def get_best_order(self) -> Optional[Order]:
        self.pop_until()
        if len(self.priority_queue.queue) > 0:
            return self.priority_queue.queue[0]
        else:
            return None

    def get_best_price(self) -> Optional[float]:
        self.pop_until()
        if len(self.priority_queue.queue) > 0:
            return self.priority_queue.queue[0].price
        else:
            return None

    def pop_until(self) -> None:
        delete_orders = [
            order
            for order in self.priority_queue.queue
            if order.is_canceled or order.is_expired(time=self.time)
        ]
        for delete_order in delete_orders:
            self.priority_queue.queue.remove(delete_order)

    def set_time(self, time: int) -> None:
        self.time = time

    def update_time(self) -> None:
        self.time += 1

    def __len__(self) -> int:
        self.pop_until()
        return len(self.priority_queue.queue)
