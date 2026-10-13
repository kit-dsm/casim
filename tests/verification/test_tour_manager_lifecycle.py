"""Status mutation boundary and generated lifecycle traces."""

import ast
from pathlib import Path

from hypothesis import settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from ware_ops_algos.algorithms import BatchObject, NodeType, Route, RouteNode, RoutingOrigin
from ware_ops_algos.planning.lifecycle import TOUR_ORDER, TOUR_TRANSITIONS, TourStates

from casim.state.tour_manager import TourManager


def test_only_tour_manager_writes_tour_status():
    root = Path(__file__).resolve().parents[2] / "src" / "casim"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if path.name == "tour_manager.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                assert not any(isinstance(target, ast.Attribute)
                               and target.attr == "status" for target in targets), (
                    path, node.lineno)


def _empty_route() -> Route:
    node = (0, 0)
    return Route(
        distance=0.0, routing_origin=RoutingOrigin(node),
        route=[node], item_sequence=[], batch=BatchObject(0, []),
        annotated_route=[RouteNode(node, NodeType.ROUTE)],
    )


class TourLifecycleMachine(RuleBasedStateMachine):
    tours = Bundle("tours")

    def __init__(self):
        super().__init__()
        self.manager = TourManager()
        self.seen: dict[int, list[TourStates]] = {}

    @rule(target=tours)
    def create(self):
        tour_id = self.manager.create_tour(_empty_route(), 0.0)
        self.seen[tour_id] = [TourStates.PLANNED]
        return tour_id

    @rule(tour_id=tours, target_status=st.sampled_from(TourStates))
    def advance(self, tour_id, target_status):
        before = self.manager.tour_status(tour_id)
        try:
            self.manager.advance(tour_id, target_status)
        except ValueError:
            assert self.manager.tour_status(tour_id) is before
            return
        after = self.manager.tour_status(tour_id)
        assert after in TOUR_TRANSITIONS[before]
        self.seen[tour_id].append(after)

    @invariant()
    def statuses_never_move_backward(self):
        for tour_id, history in self.seen.items():
            assert self.manager.tour_status(tour_id) is history[-1]
            assert all(TOUR_ORDER.index(left) < TOUR_ORDER.index(right)
                       for left, right in zip(history, history[1:]))


TestTourLifecycle = TourLifecycleMachine.TestCase
TestTourLifecycle.settings = settings(max_examples=75, stateful_step_count=40)
