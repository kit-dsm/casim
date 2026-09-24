from casim.domain_objects.sim_domain import SimWarehouseDomain

class Condition:
    def __init__(self):
        pass

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        pass

class BreakCondition(Condition):
    def __init__(self):
        super().__init__()

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if state.dynamic_warehouse_info.is_break:
            print(f"At {state.dynamic_warehouse_info.time}: break is active, skipping decision")
            return False
        else:
            return True


class DockCapacityCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        dynamic_warehouse_info = state.dynamic_warehouse_info
        active_tours = dynamic_warehouse_info.active_tours
        n_active_tours = len(active_tours)
        # n_free_pickers = len(state.resources.resources)
        if dynamic_warehouse_info.n_staged_pallets + n_active_tours > self.threshold:
            return False
        else:
            return True


class NbrPickersCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.resources.resources) >= self.threshold:
            return True
        else:
            return False


class NbrOrdersCondition(Condition):
    def __init__(self, threshold: int, allow_when_done: bool = False):
        super().__init__()
        self.threshold = threshold
        self.allow_when_done = allow_when_done

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if (len(state.orders.orders) >= self.threshold or
                self.allow_when_done and state.dynamic_warehouse_info.done
                and bool(state.orders.orders)):
            return True
        else:
            return False


class ActiveTourReadyCondition(Condition):
    """A visible candidate can be considered once the current pick has ended."""

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        dynamic = state.dynamic_warehouse_info
        return (dynamic.active_tour_id is not None
                and bool(dynamic.active_candidate_ids)
                and len(dynamic.active_tours) == 1
                and dynamic.active_tours[0].picking_until is None)

class NbrBatchesCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.dynamic_warehouse_info.buffered_batches) >= self.threshold:
            return True
        else:
            return False

