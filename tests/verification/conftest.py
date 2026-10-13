"""Full-state plan checks for scenario verification tests."""

import pytest

from ware_ops_algos.planning.invariants import check_plan, check_progress
from ware_ops_algos.planning.lifecycle import Accepts, ResourceStates, TourStates
from ware_ops_algos.planning.scope import RevisionScope, StateInformation

from casim.loggers import EventLogger
from casim.simulation_engine.state_adapter import StateAdapter


class InvariantLogger(EventLogger):
    def __init__(self):
        self.adapter = StateAdapter(
            StateInformation(orders=True, pick_lists=True,
                             tours=frozenset(TourStates),
                             resources=frozenset(ResourceStates)),
            RevisionScope(TourStates.PLANNED, Accepts.NONE),
        )
        self.previous = None

    def _check(self, sim, source="state"):
        plan = self.adapter.transform_state(
            sim.state, "invariant", "OBRSP").dynamic_warehouse_info.plan
        assert not (violations := check_plan(plan)), violations
        if self.previous is not None:
            before_by_id = {view.tour_id: view for view in self.previous.tours}
            prefix_detail = []
            for view in plan.tours:
                old = before_by_id.get(view.tour_id)
                if old is None:
                    continue
                left, right = old.pinned_prefix, view.pinned_prefix
                if right[:len(left)] != left:
                    mismatch = next((i for i, (a, b) in enumerate(zip(left, right))
                                     if a != b), min(len(left), len(right)))
                    prefix_detail.append((view.tour_id, len(left), len(right), mismatch,
                                          left[max(0, mismatch - 2):mismatch + 3],
                                          right[max(0, mismatch - 2):mismatch + 3]))
            assert not (violations := check_progress(self.previous, plan)), (
                source, violations, prefix_detail,
            )
        self.previous = plan

    def on_reset(self, sim, domain):
        self.previous = None
        self._check(sim)

    def on_event(self, event, sim):
        self._check(sim, type(event).__name__)

    def on_decision(self, problem_class, solution, state_snapshot, sim):
        self._check(sim, problem_class)

    def on_done(self, sim):
        self._check(sim)


@pytest.fixture
def invariant_logger():
    return InvariantLogger()
