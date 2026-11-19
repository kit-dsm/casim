"""A pull that routes once (C9): plan once with PlanOnlyPolicy, pull with PlannedRoutes.

OBRSP batches, routes and schedules when the orders are in and commits
nothing; ORSP keeps each planned tour's route and commits one tour to the
picker that becomes free. The run equals the pull in which one decision
routes and schedules all pick lists every time, and no route is computed after
the first decision.
"""

import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf, open_dict

from casim.run import compose_scenario, run
from tests.test_pull_assignment import FIFO, PULL, RECORDED_REPLAY

PLANNED = ["casim.pipelines.problem_based_template.InstanceLoader",
           "casim.pipelines.problem_based_template.OrdersProvider",
           "casim.pipelines.problem_based_template.PickListProvider",
           "casim.pipelines.subproblems.picker_routing.PlannedRoutes",
           "casim.pipelines.subproblems.scheduling.FIFOScheduler"]


def _route_once(tmp_path):
    cfg = compose_scenario(RECORDED_REPLAY, ["source.presence=recorded", FIFO],
                           output_dir=tmp_path)
    with open_dict(cfg):
        plan = cfg.decisions.OBRSP
        plan.state_information.tours = []
        plan.state_information.resources = ["off_shift", "idle", "reserved", "busy"]
        plan.scope.freeze_from = "planned"
        plan.commitment_policy = {
            "_target_": "casim.decision_engine.commitment_policies.PlanOnlyPolicy"}
        plan.precedence = 0
        plan.conditions = [{"_target_": "casim.simulation_engine.conditions.NbrOrdersCondition",
                            "threshold": 1}]
        cfg.decisions.ORSP = OmegaConf.create({
            "problem_class": "ORSP",
            "state_information": {"orders": False, "pick_lists": False,
                                  "tours": ["planned"], "resources": ["idle"]},
            "scope": {"freeze_from": "scheduled", "frozen_accepts": "none"},
            "solver": {"_target_": "casim.pipelines.pipeline_runner.CoSySolver",
                       "repo": {"components": PLANNED}},
            "commitment_policy": {
                "_target_": "casim.decision_engine.commitment_policies.SchedulingCommitmentPolicy",
                "n_jobs": 1},
            "precedence": 10,
            "triggers": ["OrderStreamClosed", "PickerIdle", "PickerArrival"],
            "conditions": [{"_target_": "casim.simulation_engine.conditions.NbrPickersCondition",
                            "threshold": 1}],
        })
    run(cfg)
    return tmp_path


def _tours(root: Path) -> pd.DataFrame:
    columns = ["picker_id", "status", "start", "setup_end", "end", "distance", "order_ids"]
    return (pd.read_csv(root / "facts" / "tours.csv")[columns]
            .sort_values("order_ids").reset_index(drop=True))


def test_routing_once_gives_the_pull_that_routes_every_time(tmp_path):
    every_time = tmp_path / "every_time"
    run(compose_scenario(RECORDED_REPLAY, ["source.presence=recorded", FIFO, *PULL],
                         output_dir=every_time))
    once = _route_once(tmp_path / "once")
    pd.testing.assert_frame_equal(_tours(once), _tours(every_time))
    service = [json.loads((root / "service.json").read_text()) for root in (once, every_time)]
    assert service[0]["orders"] == service[1]["orders"]
    assert service[0]["stranded"] == {"tours": 0, "orders": 0}


def test_the_pull_decision_routes_nothing(tmp_path, monkeypatch):
    from ware_ops_algos.algorithms import UShapeRouting
    calls = []
    original = UShapeRouting.solve

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(UShapeRouting, "solve", counting)
    _route_once(tmp_path)
    # One route for each of the four one-order pick lists, all computed by the
    # plan-only decision; the pull decisions that follow route nothing.
    assert len(calls) == 4


def test_plan_only_commits_nothing():
    from casim.decision_engine.commitment_policies import PlanOnlyPolicy
    assert PlanOnlyPolicy().apply((), None) == ((), None)
