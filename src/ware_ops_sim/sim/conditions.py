from ware_ops_sim.sim.sim_domain import SimWarehouseDomain


class Condition:
    def __init__(self):
        pass

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        pass


class TrueCondition(Condition):
    def __init__(self):
        super().__init__()

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return True


class PickerAvailCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.resources.resources) >= 1 >= self.threshold:
            return True
        else:
            return False


class NumberOrdersCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.orders.orders) >= self.threshold:
            return True
        else:
            print(len(state.orders.orders))
            return False


class NbrOrdersNbrPickersCondition(Condition):
    def __init__(self, order_window: int, picker_aval):
        super().__init__()
        self.order_window = order_window
        self.picker_aval = picker_aval

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if (len(state.orders.orders) >= self.order_window
                and len(state.resources.resources) >= self.picker_aval):
            return True
        else:
            return False


class ShiftStartNumbOrdersCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        # if state.dynamic_info.time > 0:
        #     return True
        if len(state.orders.orders) >= self.threshold:
            return True
        else:
            return False

# class ReoptRequirementPolicy(RequirementsPolicy):