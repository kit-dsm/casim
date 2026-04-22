from casim.domain_objects.sim_domain import SimWarehouseDomain

class Condition:
    def __init__(self):
        pass

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        pass

class DockCapacityCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        dynamic_warehouse_info = state.dynamic_warehouse_info
        if dynamic_warehouse_info.n_staged_pallets + 1 > self.threshold:
            return False
        else:
            return True


class NbrPickersCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.resources.resources) >= 1 >= self.threshold:
            return True
        else:
            return False


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
        if len(state.dynamic_warehouse_info.buffered_pick_lists) >= self.threshold:
            return True
        else:
            return False

