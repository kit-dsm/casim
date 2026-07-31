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
        return len(state.resources.resources) >= self.threshold


class NbrOrdersCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.orders.orders) >= self.threshold:
            return True
        else:
            return False

class NbrBatchesCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.dynamic_warehouse_info.buffered_batches) >= self.threshold:
            return True
        else:
            return False

