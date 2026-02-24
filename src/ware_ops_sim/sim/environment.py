import copy
import csv
import heapq
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from ware_ops_algos.algorithms.algorithm import PlanningState, RoutingSolution, AssignmentSolution, SchedulingSolution, \
    CombinedRoutingSolution, PickList, OrderSelectionSolution, AlgorithmSolution
from ware_ops_algos.algorithms import Route, SchedulingInput

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import Order, BaseWarehouseDomain, OrdersDomain, OrderType, Resource, Resources, ResourceType

from ware_ops_sim.sim import SimWarehouseDomain
from ware_ops_sim.sim.state.state_transformer import StateTransformer
from ware_ops_sim.sim.events.event_manager import EventManager
from ware_ops_sim.sim.state.tour_manager import TourStates, TourPlanningState
from ware_ops_sim.sim.state import SimulationState
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, Event, TourStart, SequencingDone, RoutingDone, AssignmentDone, \
    PickListDone, BaseTourEvent, PickerArrival, OrderSelectionDone

# from ware_sim.sim.sim_utils import logger
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WarehouseSimulation:
    """
    Manages the event queue and drives the simulation forward.

    The SimulationEngine is responsible for maintaining the heap of events,
    adding new events to the simulation, and processing events in chronological order.
    It works in conjunction with the SimulationState to update the state of the
    simulation as events are processed.

    Attributes:
        state (SimulationState): The current state of the simulation.
        events (List[Event]): A heap queue of events to be processed.
    """
    def __init__(self,
                 state_transformers: dict[str, StateTransformer],
                 control: DecisionEngine,
                 domain_cache_path: str,
                 loader_kwargs: dict,
                 data_loader: DataLoader = None,
                 order_list_path=None,
                 order_line_path=None,
                 reset_hook: Callable[['WarehouseSimulation'], None] = None,
                 track_states: bool = False):
        self.cache_path = None
        self.state_transformers = state_transformers
        self.state: SimulationState | None = None
        self.domain_cache_path = domain_cache_path
        self.control = control
        self.events = []
        self.event_manager = EventManager()
        self.order_list_path = order_list_path
        self.order_line_path = order_line_path
        self.reset_hook = reset_hook
        self.loader_kwargs = loader_kwargs
        self.n_orders = 0
        self.track_states = track_states
        if data_loader:
            self.data_loader = data_loader

    def load_data(self) -> SimWarehouseDomain:
        # domain = self.data_loader.load(self.order_list_path,
        #                                self.order_line_path)
        domain = self.data_loader.load(**self.loader_kwargs)
        return domain

    def reset(self, hooks: list[Callable[['WarehouseSimulation', SimWarehouseDomain], None]] = None):
        domain = self.load_data()
        orders = domain.orders.orders
        self.n_orders = len(orders)
        self.state = SimulationState(
            layout=domain.layout,
            articles=domain.articles,
            storage=domain.storage,
            resources=domain.resources
        )
        for order in orders:
            self.add_order(order)
        if hooks:
            for hook in hooks:
                hook(self, domain)

    def add_order(self, order: Order):
        self.add_event(OrderArrival(order.order_date, order))

    def add_pick_list(self, pl: PickList):
        self.add_event(PickListDone(pl.release, pl))

    def add_event(self, event: Event):
        heapq.heappush(self.events, event)


    def step_sim(self, solution: AlgorithmSolution, problem):
        state_transformer = self.state_transformers[problem]
        events_to_add, _, _ = self.solution_to_events(solution)
        state_transformer.on_success(self.state, solution)

        # buffer = self.state.order_manager.get_order_buffer()
        # if not self.events and len(buffer) > 0:
        #     print("check")
        #     # TODO Does this cover all possibilities?
        #     # Only works if order buffer is larger than threshold
        #     resources = self.state.resource_manager.get_resources()
        #     for r in resources.resources:
        #         print("res", r)
        #         if not r.occupied:
        #             events_to_add.append(
        #                 PickerArrival(
        #                     time=self.state.current_time,
        #                     picker_id=r.id))

        for e in events_to_add:
            self.add_event(e)

    def run(self):  #  step_no_action()
        """
         Executes the main simulation loop.

        This method runs the entire simulation by processing events in chronological order.
        It continues until all events in the queue have been handled or until an external
        decision is required.

        Returns:
            (done, context): done is True if simulation complete, False if external decision needed.
                         context contains dynamic_info when external decision required.
        """
        logger.info("Starting simulation")
        print("N_orders", self.n_orders)
        while self.events:
            event = heapq.heappop(self.events)
            # assert event.time >= self.state.current_time, f"{event.time}, {self.state.current_time}, {event}"
            # logger.debug(f"Event {event} popped at state time: {self.state.current_time}, events start: {event.time}")
            self.state.current_time = event.time
            events_to_add = event.handle(self.state)

            for e in events_to_add:
                self.add_event(e)

            if event.__class__ in self.control.triggers.keys():
                # new_events, needs_decision, dynamic_info = self.__on_trigger(event)
                problem = self.control.triggers[event.__class__]
                state_transformer = self.state_transformers[problem]

                if isinstance(event, BaseTourEvent):
                    self.state.current_picker_id = event.get_tour(self.state).assigned_resource
                elif isinstance(event, PickerArrival):
                    self.state.current_picker_id = event.picker_id
                elif isinstance(event, OrderSelectionDone):
                    self.state.current_picker_id = event.picker_id
                else:
                    print("Event", type(event))

                dynamic_info = state_transformer.transform_state(self.state,
                                                                 problem)
                # print(problem)
                condition = self.control.requirements_policies[problem]
                if condition.get_decision(dynamic_info):  # TODO make sure we do not have to check this twice
                    if (self.control.learnable_problems and
                            problem in self.control.learnable_problems):
                        return False, dynamic_info
                else:
                    print("Not condition", len(dynamic_info.resources.resources))
                print("Trigger event:", event)
                solution = self.__on_trigger(event, dynamic_info)

                if solution:
                    self.step_sim(solution, problem)



                # events_to_return, decision_required, context = self.solution_to_events(solution)
                # state_transformer.on_success(self.state, solution)

                # events_to_add += new_events
            # if events_to_add:
            #     for new_event in events_to_add:
            #         self.add_event(new_event)
            if self.track_states:
                self.state.statistics.append(copy.deepcopy({
                    "event": f"{str(event.id).zfill(5)} - {event.__class__.__name__}",
                    "time": self.state.current_time,
                    "pickers": self.state.resource_manager.get_resources()
                }))

        logger.info("Simulation complete")

        tour_history = self.state.tour_manager._picker_history
        all_orders = []
        counter = 0
        for picker_id in tour_history.keys():
            for tour_id in tour_history[picker_id]:
                logger.info(f"Picker {picker_id} made tour {tour_id}")
                tour = self.state.tour_manager.get_tour(tour_id)
                logger.info(f"Tour contained {tour.order_numbers}")
                counter += len(tour.order_numbers)
                for o_id in tour.order_numbers:
                    all_orders.append(o_id)
        logger.info(f"{len(all_orders)} picked in sum")
        pending_tours = []
        for t_id in self.state.tour_manager.all_tours.keys():
            t = self.state.tour_manager.all_tours[t_id]
            if t.status != TourStates.DONE:
                pending_tours.append(t_id)
        # print("pending_tours", pending_tours)
        # print("Resource", self.state.tour_manager.get_tour(pending_tours[0]).assigned_resource)
        print("buffer", len(self.state.order_manager.get_order_buffer()))

        tours = []
        for t_id in self.state.tour_manager.all_tours.keys():
            t = self.state.tour_manager.all_tours[t_id]
            assert t.status == TourStates.DONE
            tours.append(t)

        # processed_orders = []
        # for o_id in self.state.order_manager._order_history.keys():
        #     o = self.state.order_manager._order_history[o_id]
        #     processed_orders.append(o)
        # print(processed_orders)
        processed_orders = self.state.tracker.processed_orders
        sol = self._evaluate_due_dates(tours, processed_orders)
        print("Max makespan", sol["completion_time"].max())
        print("Final average makespan", self.state.tracker.average_tour_makespan)
        print("Final # on timem", sol["on_time"].sum())
        sol.to_csv("./solution.csv")
        self.state.tracker.final_makespan = sol["completion_time"].max()
        self.state.tracker.save_logs()
        print(sol)
        # print("tardiness", sol["tardiness"].mean())
        # print("assignable tours", self.state.tour_manager.assignable_tours)
        # with open("./logs.pkl", "wb") as f:
        #     pickle.dump(self.state.statistics, f)
        # with open('../replay.csv', 'w', newline='') as f:
        #     # Dummy results writer can be more sophisticated
        #     writer = csv.writer(f)
        #     writer.writerows(self.state.statistics)
        return True, None

    def __on_trigger(self, event: Event, dynamic_info: SimWarehouseDomain) -> AlgorithmSolution:
        """
        Trigger Algorithm Sketch

        Rough implementation of:
        on(<Trigger>)
            problem = problems[Trigger]
            S_t = getState(problem) # build problem specific state information
            d_t = decision(<algorithm>, S_t) # "decision" is 3D4L loop (+ selection) resolved into events
            S_t+1 = step(d_t, S_t)
        TriggerEvent <-> TriggerAlgorithm
        TriggerAlgorithm is executed on TriggerEvent

        Handle trigger event - either solve via configured execution or signal external decision (e.g. RL).

        Returns:
            solution: AlgorithmSolution
        """
        solution = None
        problem = self.control.triggers[event.__class__]
        runner = self.control.get_execution()

        condition = self.control.requirements_policies[problem]
        if condition.get_decision(dynamic_info):
            print(f"{condition} triggered at {self.state.current_time}")
            solution = runner.solve(dynamic_info)
            # print(f"Received: {type(solution)} solution")

        return solution

    def solution_to_events(self, solution: AlgorithmSolution):
        print("Solution type", type(solution))
        if isinstance(solution, OrderSelectionSolution):
            events_to_return = self._order_selection_to_events(solution)

        elif isinstance(solution, CombinedRoutingSolution):
            print("Resource before Routes Creation")
            # print(self.state.current_picker_id)
            events_to_return = self._routes_to_events(solution)

        elif isinstance(solution, AssignmentSolution):
            tour_id = None
            assignable_tours = self.state.tour_manager.assignable_tours
            for a in solution.assignments:
                pl_id = a.pick_list.id
                for t_id in assignable_tours:
                    if assignable_tours[t_id].pick_list.id == pl_id:
                        tour_id = t_id
                if tour_id:
                    del self.state.tour_manager.assignable_tours[tour_id]
                else:
                    raise Exception
            # TODO We somehow need to achieve this through requirements
            # if tour_id is not None:
            #     del self.state.tour_manager.assignable_tours[tour_id]
            events_to_return = self._assignment_to_events(solution)

        elif isinstance(solution, SchedulingSolution):
            events_to_return = self._schedules_to_events(solution)

        else:
            raise Exception("Not a known solution", type(solution))

        return events_to_return, False, {}

    # These functions return ProcessEvents that add solution objects to state
    def _order_selection_to_events(self, order_selection_sol: OrderSelectionSolution):
        events_to_return: list[Event] = []
        orders_selected = order_selection_sol.selected_orders
        for o in orders_selected:
            events_to_return.append(OrderSelectionDone(self.state.current_time, o, self.state.current_picker_id))
        return events_to_return

    def _schedules_to_events(self, sequencing_sol: SchedulingSolution):
        """
        Turn sequencing solution into TourStart events.
        """
        events_to_return: list[Event] = []
        sequences = sequencing_sol.jobs
        assignments = sorted(sequences, key=lambda a: (a.picker_id, a.start_time))
        # print(assignments)
        for a in assignments:
            print("picker in sequence: ", a.picker_id)
            print(f"Tour: {a.batch_idx}, start: {a.start_time}, end: {a.end_time}, distance: {a.route.distance}")
            print(f"picker speed: {self.state.resource_manager.get_resource(a.picker_id).speed}")
            events_to_return.append(SequencingDone(self.state.current_time, a))
        return events_to_return

    def _routes_to_events(self, routing_solution: CombinedRoutingSolution) -> list[RoutingDone]:
        events_to_return = []
        routes = routing_solution.routes
        # print("routes received", len(routes))
        for r in routes:
            # print("Picker ID", self.state.current_picker_id)
            events_to_return.append(RoutingDone(self.state.current_time, r, self.state.current_picker_id))
            # print("orders in solutions", r.pick_list.order_numbers)
        return events_to_return

    def _assignment_to_events(self, assignment_solution: AssignmentSolution):
        events_to_return = []
        assignments = assignment_solution.assignments
        for assignment in assignments:
            events_to_return.append(AssignmentDone(self.state.current_time,
                                    assignment))
        return events_to_return

    def _evaluate_due_dates(self, assignments: list[TourPlanningState], orders: list[Order]):
        order_by_id = {o.order_id: o for o in orders}
        records = []
        for ass in assignments:
            end_time = ass.end_time
            for on in ass.original_route.pick_list.order_numbers:
                o = order_by_id.get(on)
                if o is None:
                    continue
                if o.due_date is None:
                    due_ts = None  # skip if no due date
                    lateness = None
                    tardiness = None
                    on_time = None
                else:
                    due_ts = o.due_date #.timestamp()
                    lateness = end_time - due_ts
                    tardiness = max(0, lateness)
                    on_time = end_time <= due_ts
                arrival_time = o.order_date
                start_time = ass.start_time
                records.append({
                    "order_number": on,
                    "arrival_time": arrival_time,  # datetime.fromtimestamp(arrival_time),
                    "start_time": start_time,  # datetime.fromtimestamp(start_time),
                    "tour_id": ass.tour_id,
                    "picker_id": ass.assigned_resource,
                    "picker_type": self.state.resource_manager.get_resource(ass.assigned_resource).__class__.__name__,
                    "completion_time": end_time,  # datetime.fromtimestamp(end_time),
                    "due_date": due_ts,
                    "lateness": lateness,
                    "tardiness": tardiness,
                    "on_time": on_time,
                    "makespan": end_time - start_time
                })
        return pd.DataFrame(records)

