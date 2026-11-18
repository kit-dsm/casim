"""What each replanning policy of the IJPE plan does at a disruption (C6).

One picker has three tours A, B and C holding orders 1, 2 and 3, due at
1000, 2000 and 3000. At time 100 a truck disruption makes order 3 due at
500. The replanning decision runs through the decision engine and ``apply``
with a scheduler that orders the tours it may rebuild by earliest due date,
back to back from now; it receives exactly what the pipelines receive
(``rebuildable_tours``). The policies are those of Section 7 (P4.1) of
``docs/ijpe/plan.md``.
"""

import copy

import pytest
from ware_ops_algos.algorithms import (
    BatchObject, Job, ScheduledJob, SchedulingSolution, WarehouseOrder,
)
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.planning.lifecycle import Accepts, ResourceStates, TourStates
from ware_ops_algos.planning.scope import RevisionScope, StateInformation

from casim.decision_engine.commitment_policies import (
    CommitAllPolicy, SchedulingCommitmentPolicy,
)
from casim.decision_engine.decision_engine import DecisionEngine
from casim.decision_engine.solution_ranker import SolutionRanker
from casim.events.operational_events import TruckDisruption
from casim.pipelines.tour_filters import rebuildable_tours
from casim.run import prepare
from casim.simulation_engine.state_adapter import StateAdapter
from casim.state.apply import apply
from tests.scenario_configs import walk_config

DUE = {1: 1000, 2: 2000, 3: 3000}


class EarliestDueDate:
    def solve(self, snapshot):
        plan = snapshot.dynamic_warehouse_info.plan
        time, jobs = plan.time, []
        for view in sorted(rebuildable_tours(plan),
                           key=lambda v: (v.batch.earliest_due_date, v.tour_id)):
            duration = view.end_time - view.start_time
            job = Job(job_id=view.tour_id, processing_time=duration, release_time=0,
                      due_date=view.batch.earliest_due_date, n_picks=0,
                      route=view.route, batch=view.batch)
            jobs.append(ScheduledJob(job=job, picker_id=0, start_time=time,
                                     end_time=time + duration))
            time += duration
        return {"edd": SchedulingSolution(jobs=jobs)}


def _state(tmp_path, scheduled):
    """Tours A, B, C (orders 1, 2, 3) on picker 0; the first ``scheduled`` committed."""
    sim = prepare(walk_config(tmp_path, "cosy_repo=paper_patrol",
                              "variants=explicit_wait0")).sim
    sim.reset()
    state = sim.state
    state.resource_manager.arrive(0)
    route = all_aisles_patrol_route(state.layout_manager.get_layout())
    d = route.distance
    ids = []
    for index, order_id in enumerate((1, 2, 3)):
        order = WarehouseOrder(order_id=order_id, due_date=DUE[order_id], order_date=0)
        state.order_manager.add_order_to_history(copy.deepcopy(order))
        tour_route = copy.copy(route)
        tour_route.batch = BatchObject(order_id, [order])
        start = 1000 + index * d
        job = Job(job_id=-1, processing_time=d, release_time=0, due_date=DUE[order_id],
                  n_picks=0, route=tour_route, batch=tour_route.batch)
        ids.append(state.add_sequencing_to_planning_state(ScheduledJob(
            job=job, picker_id=0, start_time=start, end_time=start + d)))
    for tour_id in ids[:scheduled]:
        state.tour_manager.advance(tour_id, TourStates.SCHEDULED)
    state.current_time = 100
    TruckDisruption(100, order_ids=[3], due_date=500).handle(state)
    return state, dict(zip("ABC", ids))


def _replan(state, scope, commitment):
    adapter = StateAdapter(
        StateInformation(tours=frozenset({TourStates.PLANNED, TourStates.SCHEDULED}),
                         resources=frozenset({ResourceStates.IDLE, ResourceStates.RESERVED})),
        scope)
    engine = DecisionEngine({"replan": EarliestDueDate()},
                            {"replan": SolutionRanker("makespan")},
                            {"replan": commitment})
    result = engine.on_trigger(adapter.transform_state(state, "replan", "ORSP"))
    apply(state, result.commands)
    return result.commands


def _summary(state, tours):
    """Status and queue position of A, B, C; every order in exactly one live tour."""
    live = [t for t in state.tour_manager.all_tours.values()
            if t.status is not TourStates.CANCELLED]
    held = sorted(order_id for t in live for order_id in t.order_numbers)
    assert held == [1, 2, 3], "an order is in no tour or in two"
    assert sorted(t.tour_id for t in live) == sorted(tours.values()), "a tour was created"
    names = {tour_id: name for name, tour_id in tours.items()}
    return ({name: state.tour_manager.get_tour(tour_id).status.value
             for name, tour_id in tours.items()},
            "".join(names[tour_id] for tour_id in state.tour_manager.queue(0)))


def test_append_only_keeps_every_committed_tour(tmp_path):
    state, tours = _state(tmp_path, scheduled=3)
    starts = {n: state.tour_manager.get_tour(t).planned_start for n, t in tours.items()}
    commands = _replan(state, RevisionScope(TourStates.PLANNED, Accepts.NONE), CommitAllPolicy())
    assert commands == ()
    assert _summary(state, tours) == ({"A": "scheduled", "B": "scheduled", "C": "scheduled"},
                                      "ABC")
    assert {n: state.tour_manager.get_tour(t).planned_start for n, t in tours.items()} == starts


def test_replan_unstarted_revises_committed_tours_in_place(tmp_path):
    state, tours = _state(tmp_path, scheduled=3)
    _replan(state, RevisionScope(TourStates.PENDING, Accepts.NONE), CommitAllPolicy())
    assert _summary(state, tours) == ({"A": "scheduled", "B": "scheduled", "C": "scheduled"},
                                      "CAB")
    assert state.tour_manager.get_tour(tours["C"]).planned_start == 100


def test_rolling_commits_one_tour_and_keeps_the_rest_planned(tmp_path):
    # A is committed, B and C are drafts. Every unstarted tour may be revised;
    # the policy commits the earliest draft (C), and A stays committed after it.
    state, tours = _state(tmp_path, scheduled=1)
    _replan(state, RevisionScope(TourStates.PENDING, Accepts.NONE),
            SchedulingCommitmentPolicy(n_jobs=1))
    assert _summary(state, tours) == ({"A": "scheduled", "B": "planned", "C": "scheduled"},
                                      "CA")


@pytest.mark.parametrize("scheduled", [1, 3])
def test_without_a_replanning_decision_the_disruption_changes_no_tour(tmp_path, scheduled):
    state, tours = _state(tmp_path, scheduled=scheduled)
    expected = ({"A": "scheduled", "B": "scheduled" if scheduled == 3 else "planned",
                 "C": "scheduled" if scheduled == 3 else "planned"},
                "ABC" if scheduled == 3 else "A")
    assert _summary(state, tours) == expected
    assert state.tour_manager.get_tour(tours["C"]).batch.orders[0].due_date == 500
