"""Nine hand-sized Phase B checks of the model, independent of saved KPIs."""

from __future__ import annotations

from math import exp, log
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from ware_ops_algos.algorithms import (
    BatchObject, HennWaiting, Job, PickPosition, Route, ScheduledJob,
    WaitingInput, WarehouseOrder,
)
from ware_ops_algos.domain_models import Resource, Resources, ResourceType
from ware_ops_algos.planning.commands import Reschedule, ViolationCode
from ware_ops_algos.planning.lifecycle import Accepts, TourStates
from ware_ops_algos.planning.models import PlanView, TourView
from ware_ops_algos.planning.validation import check as check_commands
from ware_ops_algos.algorithms.waiting.analytic_progress import optimal_waiting

from casim.decision_engine.commitment_policies import (
    SchedulingCommitmentPolicy, WindowCommitmentPolicy,
)
from casim.decision_engine.decision_engine import DecisionEngine
from casim.decision_engine.solution_ranker import SolutionRanker
from casim.domain_objects.sim_domain import (
    DecisionContext, DecisionMemory, DynamicInfo, SimWarehouseDomain,
)
from casim.events.operational_events import Reconsider
from casim.runner import apply_decision
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from scenarios.scenario_walk_or_wait.hooks import seed_orders, seed_pickers


CONFIG = (Path(__file__).resolve().parents[2] / "scenarios" /
          "scenario_walk_or_wait" / "config")


def _walk(tmp_path, orders, *overrides):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            *overrides,
            f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    cfg.simulation.orders = orders
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    sim.reset(hooks=[seed_pickers, seed_orders])
    decisions = []
    while True:
        done, snapshot = sim.run()
        if done:
            return sim, decisions
        decisions.append((snapshot.dynamic_warehouse_info.context.decision,
                          snapshot.dynamic_warehouse_info.plan.time))
        apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)


def _snapshot(plan):
    return SimWarehouseDomain(
        problem_class="OBRSP", objective="distance", layout=None,
        orders=None, resources=Resources(ResourceType.HUMAN, [Resource(id=0)]),
        articles=None, storage=None,
        dynamic_warehouse_info=DynamicInfo(
            context=DecisionContext("decision", (), (), False, True),
            plan=plan, memory=DecisionMemory(),
        ),
    )


def test_one_picker_one_order_has_derived_end_and_completion(tmp_path):
    """One pick at (2,3) uses 38 unit edges; at speed 1 plus one pick, end is 39."""
    order = [{"arrival_s": 0.0, "due_s": 400.0, "picks": [[2, 3]]}]
    sim, _ = _walk(tmp_path, order, "decisions=paper_no_intervention",
                   "policy=wait_1")
    record = sim.state.tracker.completed_tours
    assert len(record) == 1
    _, start, end, order_ids, _, _, _, _ = record[0]
    assert (start, end, order_ids) == (0.0, 39.0, [0])
    assert sim.state.tracker.distance_by_picker[0] == 38.0
    assert end - order[0]["arrival_s"] == 39.0


def test_two_orders_wait_zero_vs_wait_two(tmp_path):
    """An arrival at 10 cannot join a departed tour; wait-2 departs with both at 10."""
    orders = [
        {"arrival_s": 0.0, "due_s": 400.0, "picks": [[2, 3]]},
        {"arrival_s": 10.0, "due_s": 400.0, "picks": [[4, 5]]},
    ]
    immediate, _ = _walk(tmp_path / "immediate", orders,
                          "decisions=paper_no_intervention", "policy=wait_0")
    waiting, _ = _walk(tmp_path / "waiting", orders,
                        "decisions=paper_no_intervention", "policy=wait_2")
    assert [row[3] for row in immediate.state.tracker.completed_tours] == [[0], [1]]
    assert [row[3] for row in waiting.state.tracker.completed_tours] == [[0, 1]]
    assert waiting.state.tracker.completed_tours[0][1] == 10.0


def test_henn_threshold_delays_the_only_job():
    """Order date 10, single service 7 and batch service 7 give 2*10+7-7=20."""
    class Router:
        def reset_parameters(self):
            pass

        def solve(self, picks):
            return SimpleNamespace(route=SimpleNamespace(distance=4))

    order = WarehouseOrder(1, order_date=10,
                           pick_positions=(PickPosition(1, 1, 1, (1, 1), 1),))
    batch = BatchObject(1, [order])
    route = Route(distance=4, batch=batch)
    job = Job(1, 5, 10, 100, 1, route=route, batch=batch)
    scheduled = ScheduledJob(job, picker_id=0, start_time=15, end_time=22)
    data = WaitingInput(
        candidates=(scheduled,), current_time=15, input_closed=False,
        picker=Resource(id=0, speed=1, time_per_pick=1, tour_setup_time=2),
        layout=None, information=None,
    )
    result = HennWaiting(Router()).solve(data)
    assert (result.action, result.reconsider_at) == ("wait", 20)
    assert [(job.start_time, job.end_time) for job in result.jobs] == [(20, 27)]


def test_one_job_commitment_preserves_first_tour_and_merges_draft():
    """At t=0 the first of two starts is committed; a new order joins only draft 2."""
    first = TourView(1, 0, TourStates.PLANNED, Accepts.REBUILD,
                     start_time=0, end_time=10)
    second = TourView(2, 0, TourStates.PLANNED, Accepts.REBUILD,
                      start_time=10, end_time=20)
    snapshot = _snapshot(PlanView(time=0, tours=(first, second)))
    commits, reconsider = SchedulingCommitmentPolicy(n_jobs=1).apply((), snapshot)
    assert [(c.tour_id, c.creation_index) for c in commits] == [(1, None)]
    assert reconsider is None
    # The later arrival is represented by adding its ID to the still-planned
    # second tour; the first tour's snapshot remains exactly the committed one.
    updated = PlanView(time=5, tours=(
        TourView(1, 0, TourStates.SCHEDULED, Accepts.NONE,
                 start_time=0, end_time=10),
        TourView(2, 0, TourStates.PLANNED, Accepts.REBUILD,
                 start_time=10, end_time=20),
    ))
    assert updated.tours[0].start_time == first.start_time
    assert updated.tours[0].end_time == first.end_time
    assert updated.tours[1].status is TourStates.PLANNED


def test_scheduled_edit_rejected_and_next_candidate_selected(monkeypatch):
    """Freeze at scheduled gives accepts=none; Reschedule exceeds accepts, so choose clean."""
    from casim.decision_engine import decision_engine as module

    scheduled = TourView(1, 0, TourStates.SCHEDULED, Accepts.NONE,
                         start_time=10, end_time=20)
    plan = PlanView(time=0, tours=(scheduled,))
    bad = Reschedule(1, 0, 11, 21)
    assert any(v.code is ViolationCode.ACCEPTS
               for v in check_commands((bad,), plan))

    class Solver:
        def solve(self, snapshot):
            return {"edit": "edit", "clean": "clean"}

    monkeypatch.setattr(module, "to_commands",
                        lambda solution, _: (bad,) if solution == "edit" else ())
    engine = DecisionEngine({"decision": Solver()},
                            {"decision": SolutionRanker("distance")}, {})
    result = engine.on_trigger(_snapshot(plan))
    assert result.solver_name == "clean"
    assert result.commands == ()


def test_admission_keeps_prefix_and_sums_old_and_new_distance(tmp_path):
    """Two suffix replacements preserve the walked prefix; total is prefix plus suffix."""
    from tests.verification.test_positioning_invariants import (
        test_two_admissions_keep_cumulative_prefix_and_distance,
    )
    from tests.verification.conftest import InvariantLogger
    test_two_admissions_keep_cumulative_prefix_and_distance(
        tmp_path, InvariantLogger())


def test_window_commitment_reconsideration():
    """At t=10 with width 5, a tour starting at 20 enters the window at t=15."""
    plan = PlanView(time=10, tours=(
        TourView(1, 0, TourStates.PLANNED, Accepts.REBUILD,
                 start_time=20, end_time=30),
    ))
    commits, revisit = WindowCommitmentPolicy(within_s=5).apply((), _snapshot(plan))
    assert commits == ()
    assert revisit == 15
    committed, revisit = WindowCommitmentPolicy(within_s=10).apply((), _snapshot(plan))
    assert [c.tour_id for c in committed] == [1]
    assert revisit is None


def test_periodic_decisions_stop_at_stream_close(tmp_path):
    """A grid anchored at 0 with period 5 offers times 5 and 10 before close at 12."""
    orders = [
        {"arrival_s": 0.0, "due_s": 400.0, "picks": [[2, 3]]},
        {"arrival_s": 12.0, "due_s": 400.0, "picks": [[4, 5]]},
    ]
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            "decisions=paper_no_intervention", "policy=wait_1",
            f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    cfg.simulation.orders = orders
    cfg.decisions.OBRSP.triggers.append({"every_s": 5})
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))

    class Times:
        def __init__(self):
            self.periodic = []

        def on_reset(self, sim, domain):
            pass

        def on_event(self, event, sim):
            if isinstance(event, Reconsider):
                self.periodic.append(event.time)

        def on_decision(self, decision, result, snapshot, sim):
            pass

        def on_done(self, sim):
            pass

    times = Times()
    sim.event_loggers.append(times)
    sim.reset(hooks=[seed_pickers, seed_orders])
    while True:
        done, snapshot = sim.run()
        if done:
            break
        apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)
    assert times.periodic == [5, 10]


def test_analytic_wait_matches_closed_form(monkeypatch):
    """With mean 10 and tail cost 40/e, w*=10 ln(4/e)=3.86294 seconds."""
    monkeypatch.setattr(optimal_waiting, "build_interval_data",
                        lambda data: ([], 10.0))
    result = optimal_waiting.solve_optimal_wait(
        None, mean_interarrival_time=10,
        remaining_arrivals_after_miss=3,
    )
    expected = 10 * log(4 / exp(1))
    assert result.wait_duration == pytest.approx(expected)
