import pytest

from ware_ops_algos.algorithms import Job, ScheduledJob
from ware_ops_algos.domain_models import Resource, ResourceType, Resources
from ware_ops_algos.planning.commands import CreateTour, Reschedule
from ware_ops_algos.planning.lifecycle import Accepts, TourStates
from ware_ops_algos.planning.models import PlanView, TourView

from casim.decision_engine.commitment_policies import (
    CommitAllPolicy, SchedulingCommitmentPolicy, WindowCommitmentPolicy,
)
from casim.domain_objects.sim_domain import (
    DecisionContext, DecisionMemory, DynamicInfo, SimWarehouseDomain,
)


def _snapshot(plan, waiting=False):
    dynamic = DynamicInfo(
        context=DecisionContext("decision", (), (), False, True),
        plan=plan,
        memory=DecisionMemory(waiting=waiting),
    )
    return SimWarehouseDomain(
        problem_class="OBRSP", objective="Distance", layout=None,
        articles=None, orders=None,
        resources=Resources(ResourceType.HUMAN, [Resource(id=0)]), storage=None,
        dynamic_warehouse_info=dynamic,
    )


def _plan():
    return PlanView(
        time=10,
        tours=(
            TourView(1, 0, TourStates.PLANNED, Accepts.REBUILD,
                     start_time=20, end_time=30),
            TourView(2, 1, TourStates.PLANNED, Accepts.REBUILD,
                     start_time=11, end_time=21),
            TourView(3, 0, TourStates.SCHEDULED, Accepts.REBUILD,
                     start_time=12, end_time=22),
        ),
    )


def test_commitment_only_sees_planned_tours_of_visible_pickers():
    snapshot = _snapshot(_plan())
    assert tuple(commit.tour_id for commit in
                 CommitAllPolicy().apply((), snapshot)[0]) == (1,)
    assert SchedulingCommitmentPolicy(1).apply((), snapshot)[0][0].tour_id == 1
    assert CommitAllPolicy().apply((), _snapshot(_plan(), waiting=True)) == ((), None)


def test_window_uses_ready_time_and_reconsiders_next_tour():
    snapshot = _snapshot(_plan())
    policy = WindowCommitmentPolicy(within_s=5)
    assert policy.apply((), snapshot) == ((), 15)
    assert tuple(c.tour_id for c in WindowCommitmentPolicy(10).apply(
        (), snapshot)[0]) == (1,)


def test_created_and_rescheduled_tours_are_considered():
    snapshot = _snapshot(_plan())
    job = ScheduledJob(Job(7, 1, 0, 100, 0), picker_id=0,
                       start_time=13, end_time=14)
    commits, _ = CommitAllPolicy().apply((
        CreateTour(job), Reschedule(1, 0, 12, 22)), snapshot)
    assert [(c.tour_id, c.creation_index) for c in commits] == [
        (1, None), (None, 0),
    ]
    with pytest.raises(ValueError, match="n_jobs"):
        SchedulingCommitmentPolicy(0)
    with pytest.raises(ValueError, match="within_s"):
        WindowCommitmentPolicy(-1)
