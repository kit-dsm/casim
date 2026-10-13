import copy

from ware_ops_algos.algorithms import BatchObject
from ware_ops_algos.domain_models import Order
from ware_ops_algos.domain_models.lifecycle import OrderStates, PickListStates, TourStates


class OrderManager:
    def __init__(self):
        self.completed_orders: list[Order] = []
        self._order_buffer: dict[int, Order] = {}
        self._pick_list_buffer: list[BatchObject] = []
        self._pick_list_assignments: dict[int, BatchObject | None] = {}
        self._order_history: dict[int, Order] = {}

    def add_order_to_buffer(self, order: Order) -> None:
        self._order_buffer[order.order_id] = order

    def add_pick_list_to_buffer(self, pick_list: BatchObject) -> None:
        self._pick_list_buffer.append(pick_list)

    def add_selected_pick_list(self, pick_list: BatchObject,
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

    def order_status(self, order_id: int, tour_manager) -> OrderStates:
        """Derive an order's one lifecycle status from the live collections."""
        statuses: set[OrderStates] = set()
        if order_id in self._order_buffer:
            statuses.add(OrderStates.BUFFERED)
        if any(order_id in batch.order_numbers for batch in self._pick_list_buffer):
            statuses.add(OrderStates.BATCHED)
        if any(batch is not None and order_id in batch.order_numbers
               for batch in self._pick_list_assignments.values()):
            statuses.add(OrderStates.BATCHED)
        for tour in tour_manager.all_tours.values():
            if order_id not in tour.order_numbers or tour.status == TourStates.CANCELLED:
                continue
            if tour.status == TourStates.DONE:
                statuses.add(OrderStates.DONE)
            elif tour.status == TourStates.STARTED:
                statuses.add(OrderStates.IN_PROGRESS)
            else:
                statuses.add(OrderStates.PLANNED)
        if any(order.order_id == order_id for order in self.completed_orders):
            statuses.add(OrderStates.DONE)
        if len(statuses) != 1:
            raise ValueError(f"Order {order_id} has ambiguous lifecycle status: {statuses}")
        return next(iter(statuses))

    def pick_list_status(self, pick_list: BatchObject, tour_manager) -> PickListStates:
        """Derive a pick list's status without storing a second status flag."""
        if pick_list in self._pick_list_buffer:
            return PickListStates.BUFFERED
        for tour in tour_manager.all_tours.values():
            if tour.batch is not pick_list and tour.batch != pick_list:
                continue
            if tour.status == TourStates.DONE:
                return PickListStates.DONE
            if tour.status == TourStates.STARTED:
                return PickListStates.IN_PROGRESS
            if tour.status != TourStates.CANCELLED:
                return PickListStates.PLANNED
        raise ValueError("Pick list has no lifecycle status")

    def get_order_buffer(self) -> list[Order]:
        return list(self._order_buffer.values())

    def get_pick_list_buffer(self) -> list[BatchObject]:
        return list(self._pick_list_buffer)

    def get_selected_pick_list(self, picker_id: int) -> BatchObject:
        pl = self._pick_list_assignments[picker_id]
        self._pick_list_assignments[picker_id] = None

        return pl

    def clear_order_buffer_by_ids(self, order_ids: list[int]) -> None:
        for o_id in order_ids:
            order = self._order_buffer.pop(o_id, None)
            try:
                assert isinstance(order, Order)
            except AssertionError:
                print(f"Order with id {o_id} not found in buffer, cannot clear")
            if order is not None:
                self.add_order_to_history(order)

    def clear_order_buffer(self, orders: list[Order] | None = None) -> None:
        if orders is None:
            ids_to_clear = list(self._order_buffer.keys())
        else:
            ids_to_clear = [o.order_id for o in orders]

        for o_id in ids_to_clear:
            order = self._order_buffer.pop(o_id, None)
            try:
                assert isinstance(order, Order)
            except AssertionError:
                print(f"Order with id {o_id} not found in buffer, cannot clear")
            if order is not None:
                self.add_order_to_history(order)

    def clear_pick_list_buffer(self, pls: list[BatchObject] | None = None) -> None:

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
