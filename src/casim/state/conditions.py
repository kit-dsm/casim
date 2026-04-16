from hydra.utils import instantiate
from omegaconf import DictConfig

from casim.domain_objects.sim_domain import SimWarehouseDomain


def build_condition_policies(cfg: DictConfig) -> dict:
    return {
        policy_key: instantiate(condition_cfg)
        for policy_key, condition_cfg in cfg.decision_engine.conditions.items()
    }


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
            return False

class NumberPickListCondition(Condition):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        if len(state.dynamic_warehouse_info.buffered_pick_lists) >= self.threshold:
            return True
        else:
            return False

class NbrOrdersNbrPickersCondition(Condition):
    def __init__(self, order_window: int, picker_aval):
        super().__init__()
        self.order_window = order_window
        self.picker_aval = picker_aval

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        # if (state.warehouse_info.done
        #         and len(state.resources.resources) >= self.picker_aval):
        #     return True

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


# class BatchFullPickAvalCondition(Condition):
#     def __init__(self, picker_aval: int, n_batches: int):
#         super().__init__()
#         self.picker_aval = picker_aval
#         self.n_batches = n_batches
#
#     def get_decision(self, state: SimWarehouseDomain) -> bool:
#
#         if len(state.resources.resources) >= self.picker_aval:
#             buffered_pls: list[PickList] = state.warehouse_info.pick_list_buffer
#
#             capacity_checker = CapacityChecker(
#                 articles=state.articles,
#                 pick_cart=state.resources.resources[0].pick_cart)
#
#             for pl in buffered_pls:
#                 capacity_checker.can_add_order()
#             return True

