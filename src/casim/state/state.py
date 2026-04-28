import copy
from copy import deepcopy

from ware_ops_algos.algorithms import Route, Job, PickerAssignment, WarehouseOrder, BatchObject, PickList
from ware_ops_algos.domain_models import LayoutData, Articles, StorageLocations, Resources, Order, Resource

from .order_manager import OrderManager
from .resource_manager import ResourceManager
from .tour_manager import TourManager
from .layout_manager import LayoutManager
from .storage_manager import StorageManager
from ..trackers import ExperimentTracker


class State:
    """
    Holds all mutable simulation data.
    """
    _SHARED_FIELDS = {'_layout', '_articles', '_storage', '_resources'}

    def __init__(self,
                 layout: LayoutData,
                 articles: Articles,
                 storage: StorageLocations,
                 resources: Resources):
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
        self.tracker = ExperimentTracker(
            n_pickers=len(resources.resources),
            )
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
