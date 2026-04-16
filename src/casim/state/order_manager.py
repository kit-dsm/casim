import copy

from ware_ops_algos.algorithms import BatchObject, PickList
from ware_ops_algos.domain_models import Order


class OrderManager:
    def __init__(self):
        self.completed_orders: list[Order] = []
        self._order_buffer: dict[int, Order] = {}
        self._pick_list_buffer: list[PickList] = []
        self._pick_list_assignments: dict[int, PickList | None] = {}
        self._order_history: dict[int, Order] = {}

    def add_order_to_buffer(self, order: Order) -> None:
        self._order_buffer[order.order_id] = order

    def add_pick_list_to_buffer(self, pick_list: PickList) -> None:
        self._pick_list_buffer.append(pick_list)

    def add_selected_pick_list(self, pick_list: PickList,
                               picker_id: int) -> None:
        self._pick_list_assignments[picker_id] = pick_list

    def add_order_to_history(self, order: Order) -> None:
        o_id = order.order_id
        if o_id not in self._order_history.keys():
            self._order_history[o_id] = order

    def get_order_from_history(self, o_id: int) -> Order:
        try:
            return self._order_history[o_id]
        except KeyError:
            raise ValueError(f"Order {o_id} not found in history")

    def get_order_buffer(self) -> list[Order]:
        return list(self._order_buffer.values())

    def get_pick_list_buffer(self) -> list[PickList]:
        return list(self._pick_list_buffer)

    def get_selected_pick_list(self, picker_id: int) -> PickList:
        pl = self._pick_list_assignments[picker_id]
        self._pick_list_assignments[picker_id] = None

        return pl

    def clear_order_buffer(self, orders: list[Order] | None = None) -> None:
        if orders is None:
            ids_to_clear = list(self._order_buffer.keys())
        else:
            ids_to_clear = [o.order_id for o in orders]

        for o_id in ids_to_clear:
            order = self._order_buffer.pop(o_id, None)
            if order is not None:
                self.add_order_to_history(order)

    def clear_pick_list_buffer(self, pls: list[PickList] | None = None) -> None:

        if pls is None:
            pls_to_clear = self.get_pick_list_buffer()
        else:
            pls_to_clear = pls
        for pl in pls_to_clear:
            if pl in self._pick_list_buffer:
                self._pick_list_buffer.remove(pl)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_order_history':
                setattr(result, k, v)  # shared ref, completed orders never mutate
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result