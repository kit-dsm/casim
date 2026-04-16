import heapq

from .events import OrderArrival, Event
from ware_ops_algos.domain_models import Order


class EventManager:
    def __init__(self):
        self.events = []

    def add_order(self, order: Order):
        self.add_event(OrderArrival(order.order_date, order))

    def add_event(self, event: Event):
        heapq.heappush(self.events, event)

