from ware_ops_algos.domain_models import Order


class OrderManager:
    def __init__(self):
        self.completed_orders: list[Order] = []
        self._order_buffer: list[Order] = []
        self._order_history: dict[int, Order] = {}

    def add_order_to_buffer(self, order: Order) -> None:
        self._order_buffer.append(order)

    def add_order_to_history(self, order: Order) -> None:
        o_id = order.order_id
        if o_id not in self._order_history.keys():
            self._order_history[o_id] = order

    def get_order_from_history(self, o_id: int) -> Order:
        if o_id in self._order_history.keys():
            return self._order_history[o_id]
        else:
            raise ValueError

    def get_order_buffer(self) -> list[Order]:
        return self._order_buffer

    def clear_order_buffer(self, orders: list[Order] | None = None) -> None:
        """
        Orders are deleted from the order buffer and added to the order history
        """
        if orders is None:
            orders_to_clear = self.get_order_buffer()
        else:
            orders_to_clear = orders
        #     print("to clear", orders_to_clear)
        # print("order buffer", len(self._order_buffer))
        # for o in self._order_buffer:
        #     print(o.order_id)
        for o in orders_to_clear:
            self.add_order_to_history(o)
            # assert o in self._order_buffer
            for ob in self._order_buffer:
                if ob.order_id == o.order_id:
                    self._order_buffer.remove(ob)
