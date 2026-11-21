"""travel: pick walks to the next pick node in one event, with edge-level results.

Edge travel has one TravelEvent and one NodeArrival per graph edge. Pick
travel walks every edge up to the next pick node (or the route's end) in one
event, adding arrival times and distances edge by edge, and answers a
decision's view of a tour inside a leg with the edge walked at that time
(``TourPlanningState.edge_at``). Tours, picks, orders, KPIs, the resource log,
the audit's legs and every decision's view of the tours must be equal; only
the number of events falls.
"""

import json
from types import SimpleNamespace

import pytest
from ware_ops_algos.planning import Accepts, TourStates

from casim.domain_objects.tour_model import TourPlanningState
from casim.events.operational_events import NodeArrival, PickerIdle
from casim.run import check_travel, compose_scenario, run
from casim.simulation_engine.state_adapter import StateAdapter
from tests.test_pull_assignment import FIFO, PULL, PUSH, RECORDED_REPLAY

TRANSFORM_STATE = StateAdapter.transform_state
OUTPUTS = ("facts/tours.csv", "facts/picks.csv", "facts/orders.csv",
           "kpis.json", "tracker.json", "service.json")


def _run(tmp_path, monkeypatch, overrides, travel):
    # Every decision's view of the tours (the audit's views at each timestamp
    # differ in number, since edge travel has more timestamps).
    views = []

    def recorded(self, state, decision, problem_class):
        snapshot = TRANSFORM_STATE(self, state, decision, problem_class)
        for view in snapshot.dynamic_warehouse_info.plan.tours:
            progress = view.progress
            views.append((state.current_time, view.tour_id, None if progress is None else (
                progress.cursor, tuple(node.position for node in progress.pinned_prefix),
                progress.remaining_route, progress.routing_origin)))
        return snapshot

    monkeypatch.setattr(StateAdapter, "transform_state", recorded)
    cfg = compose_scenario(RECORDED_REPLAY, [*overrides, f"++travel={travel}"],
                           output_dir=tmp_path)
    cfg.experiment.audit = True
    result = run(cfg)
    files = {name: (tmp_path / name).read_text() for name in OUTPUTS}
    audit = json.loads((tmp_path / "audit.json").read_text())
    return result.event_count, files, audit["route_legs"], audit["violations"], views


@pytest.mark.parametrize("overrides", [
    [],
    ["source.presence=recorded", FIFO, *PULL],
    ["source.presence=recorded", FIFO, *PUSH],
])
def test_pick_travel_gives_the_results_of_edge_travel(tmp_path, monkeypatch, overrides):
    edge = _run(tmp_path / "edge", monkeypatch, overrides, "edge")
    pick = _run(tmp_path / "pick", monkeypatch, overrides, "pick")
    assert pick[0] < edge[0]
    assert pick[1] == edge[1]
    assert pick[2] == edge[2]
    assert pick[3] == edge[3] == {"check_plan": {}, "check_progress": {},
                                  "check_resources": {}}
    assert pick[4] == edge[4]


def _setup(triggers=(), tours=(), accepts=Accepts.NONE):
    scope = SimpleNamespace(accepts=lambda status: accepts)
    return SimpleNamespace(trigger_types=tuple(triggers), config=SimpleNamespace(
        state_information=SimpleNamespace(tours=frozenset(tours)), scope=scope))


def test_pick_travel_is_refused_where_nodes_in_between_matter():
    check_travel("pick", {"pull": _setup((PickerIdle,), (TourStates.STARTED,))})
    check_travel("edge", {"node": _setup((NodeArrival,))})
    with pytest.raises(ValueError, match="no event per edge"):
        check_travel("pick", {"node": _setup((NodeArrival,))})
    with pytest.raises(ValueError, match="started tour"):
        check_travel("pick", {"admit": _setup((), (TourStates.STARTED,), Accepts.REBUILD)})
    with pytest.raises(ValueError, match="edge or pick"):
        check_travel("node", {})


def test_inside_a_leg_the_tour_reports_the_edge_walked_then():
    # A leg of three edges from the cursor at node 1: lengths 2, 4 and 2 at speed 1.
    nodes = [SimpleNamespace(position=(x, 0)) for x in (0, 1, 3, 7, 9)]
    tour = TourPlanningState(tour_id=1, order_numbers=[], original_route=None, route=None,
                             batch=None, annotated_route=nodes, cursor=1)
    tour.edge_origin, tour.edge_destination = nodes[1], nodes[4]
    tour.edge_start_time, tour.edge_end_time = 10.0, 18.0
    tour.leg_ends, tour.leg_distances = [12.0, 16.0, 18.0], [2.0, 4.0, 2.0]

    assert tour.edge_at(11.0) == (0, nodes[1], nodes[2], 10.0, 12.0, 2.0)
    # At an inner node's arrival the next edge has begun, as in edge travel.
    assert tour.edge_at(12.0) == (1, nodes[2], nodes[3], 12.0, 16.0, 4.0)
    assert tour.position_at(14.0) == (5.0, 0.0)
    assert tour.edge_at(18.0) == (2, nodes[3], nodes[4], 16.0, 18.0, 2.0)
    assert tour.pinned_prefix(True, 1) == tuple(nodes[:3])
    tour.edge_origin = None
    assert tour.edge_at(14.0) is None and tour.position_at(14.0) == (1, 0)
