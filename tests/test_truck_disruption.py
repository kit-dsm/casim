"""A truck disruption changes due dates only; the plan is revised by decisions (C1).

The state holds one order in each place an open order can be: the order
buffer, the pick-list buffer, a planned tour and a scheduled tour. Tours hold
their own copies of orders, as the batches a decision returns do. Every
order starts with due date 1000; the disruption moves orders 1 to 4 to 600.
"""

import copy

from ware_ops_algos.algorithms import BatchObject, Job, ScheduledJob, WarehouseOrder
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.planning.lifecycle import Accepts, TourStates
from ware_ops_algos.planning.scope import RevisionScope, StateInformation

from casim.events.operational_events import TruckDisruption
from casim.run import prepare
from casim.simulation_engine.state_adapter import StateAdapter
from tests.scenario_configs import walk_config


def _order(order_id):
    return WarehouseOrder(order_id=order_id, due_date=1000, order_date=0)


def _state(tmp_path):
    sim = prepare(walk_config(tmp_path, "cosy_repo=paper_patrol",
                              "variants=explicit_wait0")).sim
    sim.reset()
    state = sim.state
    state.resource_manager.arrive(0)
    route = all_aisles_patrol_route(state.layout_manager.get_layout())
    state.order_manager.add_order_to_buffer(_order(1))
    state.order_manager.add_pick_list_to_buffer(BatchObject(10, [_order(2)]))
    tours = {}
    for order_id, start in ((3, 500), (4, 900)):
        state.order_manager.add_order_to_history(_order(order_id))
        batch = BatchObject(order_id, [copy.deepcopy(_order(order_id))])
        tour_route = copy.copy(route)
        tour_route.batch = batch
        job = Job(job_id=-1, processing_time=route.distance, release_time=0,
                  due_date=1000, n_picks=0, route=tour_route, batch=batch)
        tours[order_id] = state.add_sequencing_to_planning_state(ScheduledJob(
            job=job, picker_id=0, start_time=start, end_time=start + route.distance))
    state.tour_manager.advance(tours[4], TourStates.SCHEDULED)
    return state, tours


def _due_dates(state):
    orders = [*state.order_manager.get_order_buffer(),
              *(o for b in state.order_manager.get_pick_list_buffer() for o in b.orders),
              *(o for t in state.tour_manager.all_tours.values() for o in t.batch.orders)]
    return {o.order_id: o.due_date for o in orders}


def test_disruption_moves_due_dates_and_cancels_nothing(tmp_path):
    state, tours = _state(tmp_path)
    version = state.information_version
    events = TruckDisruption(100, order_ids=[1, 2, 3, 4, 99], due_date=600).handle(state)
    assert events == []
    assert _due_dates(state) == {1: 600, 2: 600, 3: 600, 4: 600}
    assert state.order_manager.get_order_from_history(3).due_date == 600
    assert state.tour_manager.get_tour(tours[3]).status is TourStates.PLANNED
    assert state.tour_manager.get_tour(tours[4]).status is TourStates.SCHEDULED
    assert state.information_version == version + 1
    assert sorted(state.tracker.shifted_orders) == [(i, 1000, 600) for i in (1, 2, 3, 4)]


def test_disruption_leaves_other_orders_alone(tmp_path):
    state, _ = _state(tmp_path)
    TruckDisruption(100, order_ids=[3], due_date=600).handle(state)
    assert _due_dates(state) == {1: 1000, 2: 1000, 3: 600, 4: 1000}


def test_next_snapshot_shows_the_new_due_date(tmp_path):
    state, tours = _state(tmp_path)
    adapter = StateAdapter(StateInformation(tours=frozenset({TourStates.PLANNED})),
                           RevisionScope(TourStates.PLANNED, Accepts.NONE))

    def snapshot_due():
        plan = adapter.transform_state(state, "replan", "OBRSP").dynamic_warehouse_info.plan
        (tour,) = [t for t in plan.tours if t.tour_id == tours[3]]
        return tour.batch.orders[0].due_date

    assert snapshot_due() == 1000
    TruckDisruption(100, order_ids=[3], due_date=600).handle(state)
    assert snapshot_due() == 600
