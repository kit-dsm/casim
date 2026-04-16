import copy
from copy import deepcopy

from ware_ops_algos.algorithms import Route, Job, PickerAssignment, WarehouseOrder, BatchObject, PickList
from ware_ops_algos.domain_models import LayoutData, Articles, StorageLocations, Resources, Order, Resource

from .order_manager import OrderManager
from .resource_manager import ResourceManager
from .tour_manager import TourManager
from .layout_manager import LayoutManager
from .storage_manager import StorageManager


class ExperimentTracker:
    def __init__(self):
        self.finished_tours = 0
        self.average_pick_time = 0
        self.average_travel_time = 0
        self.tour_makespans = []
        self.average_tour_makespan = 0
        self.final_makespan = None
        self.processed_orders = []
        self.logs: list[dict] = []

    def update_on_travel(self):
        pass

    def update_on_pick_start(self, log_entry: dict):
        self.logs.append(log_entry)

    def update_on_pick_end(self, log_entry: dict):
        self.logs.append(log_entry)

    def update_on_tour_end(self, tour_start: float, tour_finish: float, order_manager: OrderManager):
        self.finished_tours += 1
        tour_makespan = tour_finish - tour_start
        self.tour_makespans.append(tour_makespan)
        self.average_tour_makespan = sum(self.tour_makespans) / self.finished_tours

        for o_id in order_manager._order_history.keys():
            o = order_manager._order_history[o_id]
            self.processed_orders.append(o)

    def to_dataframe(self):
        """Export logs to pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame(self.logs)

    def save_logs(self):
        df = self.to_dataframe()
        df.to_csv("./experiment_logs.csv")

    def __deepcopy__(self, memo):
        result = ExperimentTracker.__new__(ExperimentTracker)
        memo[id(self)] = result
        result.finished_tours = self.finished_tours
        result.average_pick_time = self.average_pick_time
        result.average_travel_time = self.average_travel_time
        result.tour_makespans = self.tour_makespans.copy()  # list of floats, needs copy
        result.average_tour_makespan = self.average_tour_makespan
        result.final_makespan = self.final_makespan
        result.processed_orders = self.processed_orders  # shared ref
        result.logs = []  # skip entirely
        return result


class State:
    """
    Holds all mutable simulation data.
    """
    _SHARED_FIELDS = {'_layout', '_articles', '_storage', '_resources'}
    def __init__(self, layout: LayoutData, articles: Articles, storage: StorageLocations, resources: Resources):
        # time is a float (simulation time units)
        self.current_time: float = 0.0
        self.current_picker_id = None
        self.is_break: bool = False
        self.break_duration: int | None = None
        self.resource_manager = ResourceManager(resources=resources)
        self.storage_manager = StorageManager(articles=articles,
                                              storage=storage)
        self.order_manager = OrderManager()
        self.tour_manager = TourManager()
        self.layout_manager = LayoutManager(layout=layout)

        self._layout = layout
        self._articles = articles
        self._storage = storage
        self._resources = resources
        self.tracker = ExperimentTracker()
        self.statistics = []
        self.done_flag = False

    def get_storage(self) -> StorageLocations:
        return self._storage

    def available_for_planning(self, picker_id: int) -> bool:
        """Picker is usable by the planner only if not occupied and not reserved by queued tours."""
        res = self.resource_manager.get_resource(picker_id)
        return (not res.occupied) and (not self.tour_manager.has_future_tours(picker_id))

    def add_statistic(self, picker_id: int,
                      time_value: float,
                      order_id: int) -> None:
        self.statistics.append([picker_id, time_value, order_id])

    def add_selected_order_to_planning_state(self,
                                             order: WarehouseOrder,
                                             picker_id: int | None = None):
        self.tour_manager.add_selected_order(order,
                                             picker_id)

    def add_ia_to_planning_state(self, resolved_orders):
        # TODO What to do here?
        pass

    def add_pick_list_to_planning_state(self, pick_list: PickList) -> None:
        self.order_manager.add_pick_list_to_buffer(pick_list)

    def add_selected_pick_list_to_planning_state(self, pick_list: PickList,
                                                 picker_id: int) -> None:
        self.order_manager.add_selected_pick_list(pick_list,
                                                  picker_id)

    # -> All of these functions fill the TourExecution Dataclass
    # which is maintained by the tour manager
    def add_route_to_planning_state(self, route: Route, picker_id: int | None = None) -> None:
        # route: Route = deepcopy(route)
        tour_id = self.tour_manager.create_tour(route)
        assert picker_id >= 0
        if picker_id is not None:
            self.tour_manager.assign_tour(tour_id, picker_id)
            # print(f"created tour {tour_id} for picker {picker_id}")
            # print("check resource", self.tour_manager.get_tour(tour_id).assigned_resource)

        # print("Tour created:", self.tour_manager.get_tour(tour_id))
        # self.tour_manager.assign_tour(tour_id, sequencing.picker_id)

    def add_sequencing_to_planning_state(self, sequencing: Job) -> None:
        tour_id = self.tour_manager.create_tour(deepcopy(sequencing.route))
        self.tour_manager.assign_tour(tour_id, sequencing.picker_id)
        self.tour_manager.schedule_tour(tour_id,
                                        sequencing.start_time,
                                        sequencing.end_time)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._SHARED_FIELDS:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result
