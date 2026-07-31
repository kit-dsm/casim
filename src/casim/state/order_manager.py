import copy

from ware_ops_algos.algorithms import BatchObject
from ware_ops_algos.domain_models import Order


class OrderManager:
    def __init__(self):
        self.completed_orders: list[Order] = []
        self._unreleased_orders: dict[object, Order] = {}
        self._release_versions: dict[object, int] = {}
        self._order_buffer: dict[int, Order] = {}
        self._pick_list_buffer: list[BatchObject] = []
        self._pick_list_assignments: dict[int, BatchObject | None] = {}
        self._order_history: dict[int, Order] = {}
        self._next_batch_id: int = 0

    def register_order(self, order: Order) -> int:
        order_id = order.order_id
        if (
            order_id in self._unreleased_orders
            or order_id in self._order_buffer
            or order_id in self._order_history
        ):
            raise ValueError(f"Order {order_id} is already registered")
        self._unreleased_orders[order_id] = order
        self._release_versions[order_id] = 0
        return 0

    def release_order(self, order_id, expected_version: int) -> Order | None:
        if self._release_versions.get(order_id) != expected_version:
            return None
        order = self._unreleased_orders.pop(order_id, None)
        if order is None:
            return None
        self._order_buffer[order_id] = order
        return order

    def reschedule_unreleased_orders(
        self,
        *,
        from_due: float,
        new_due: float,
        new_release: float,
        max_orders: int | None = None,
    ) -> list[tuple[object, float, int]]:
        candidates = sorted(
            (
                order
                for order in self._unreleased_orders.values()
                if order.due_date is not None and order.due_date >= from_due
            ),
            key=lambda order: (
                order.due_date,
                order.order_date,
                str(order.order_id),
            ),
        )
        if max_orders is not None:
            candidates = candidates[: int(max_orders)]
        scheduled = []
        for order in candidates:
            order.order_date = float(new_release)
            order.due_date = float(new_due)
            version = self._release_versions[order.order_id] + 1
            self._release_versions[order.order_id] = version
            scheduled.append((order.order_id, float(new_release), version))
        return scheduled

    def unreleased_order_ids(self) -> set:
        return set(self._unreleased_orders)

    def add_order_to_buffer(self, order: Order) -> None:
        if order.order_id in self._order_buffer:
            raise ValueError(f"Order {order.order_id} is already buffered")
        self._order_buffer[order.order_id] = order

    def add_pick_list_to_buffer(self, pick_list: BatchObject) -> None:
        existing = self.buffered_batch_ids()
        if pick_list.batch_id in existing:
            while self._next_batch_id in existing:
                self._next_batch_id += 1
            pick_list.batch_id = self._next_batch_id
        self._next_batch_id = max(
            self._next_batch_id,
            int(pick_list.batch_id) + 1,
        )
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

    def order_history_ids(self) -> set[int]:
        return set(self._order_history)

    def mark_orders_completed(self, order_ids: list[int]) -> None:
        completed_ids = {order.order_id for order in self.completed_orders}
        for order_id in order_ids:
            if order_id in completed_ids:
                raise ValueError(f"Order {order_id} completed more than once")
            order = self.get_order_from_history(order_id)
            self.completed_orders.append(order)
            completed_ids.add(order_id)

    def get_order_buffer(self) -> list[Order]:
        return list(self._order_buffer.values())

    def planning_order_buffer(self) -> list[Order]:
        return copy.deepcopy(list(self._order_buffer.values()))

    def get_pick_list_buffer(self) -> list[BatchObject]:
        return list(self._pick_list_buffer)

    def buffered_order_ids(self) -> set[int]:
        return set(self._order_buffer)

    def buffered_batch_ids(self) -> set[int]:
        return {batch.batch_id for batch in self._pick_list_buffer}

    def get_buffered_batches_by_ids(
        self,
        batch_ids: set[int],
    ) -> list[BatchObject]:
        return [
            batch
            for batch in self._pick_list_buffer
            if batch.batch_id in batch_ids
        ]

    def planning_pick_list_buffer(self) -> list[BatchObject]:
        return copy.deepcopy(self._pick_list_buffer)

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

    def commit_order_ids(self, order_ids: list[int]) -> None:
        """Move buffered orders to history, accepting already committed IDs."""
        missing = [
            order_id
            for order_id in order_ids
            if order_id not in self._order_buffer
            and order_id not in self._order_history
        ]
        if missing:
            raise ValueError(
                "Committed orders are neither buffered nor present in order "
                f"history: {missing}"
            )
        for order_id in order_ids:
            order = self._order_buffer.pop(order_id, None)
            if order is not None:
                self.add_order_to_history(order)

    def restore_buffered_orders(self, orders: list[Order]) -> None:
        """Rollback helper for a failed multi-manager commitment."""
        for order in orders:
            self._order_history.pop(order.order_id, None)
            self._order_buffer[order.order_id] = order

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

    def remove_pick_lists_by_ids(
        self,
        batch_ids: set[int],
    ) -> list[BatchObject]:
        selected = self.get_buffered_batches_by_ids(batch_ids)
        if {batch.batch_id for batch in selected} != batch_ids:
            missing = sorted(
                batch_ids - {batch.batch_id for batch in selected}
            )
            raise ValueError(f"Batches are no longer buffered: {missing}")
        self._pick_list_buffer = [
            batch
            for batch in self._pick_list_buffer
            if batch.batch_id not in batch_ids
        ]
        return selected

    def restore_pick_lists(self, batches: list[BatchObject]) -> None:
        existing = self.buffered_batch_ids()
        for batch in batches:
            if batch.batch_id not in existing:
                self._pick_list_buffer.append(batch)
                existing.add(batch.batch_id)

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
