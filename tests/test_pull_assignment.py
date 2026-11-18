"""Push and pull assignment (casim/decision_engine/commitment_policies.py).

Push commits tours into picker queues ahead of time; pull lets a free picker
take its next tour. In recorded_replay with recorded presence, picker 1 is
present from 35 s to 50 s and picker 0 is away from 30 s to about 95 s. A
FIFO push at the start queues tour 3 with picker 1, who leaves before it can
start it; with pull, picker 0 takes it when it is back.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from ware_ops_algos.planning.commands import CreateTour

from casim.config.schema import DecisionConfig
from casim.decision_engine.commitment_policies import PullCommitmentPolicy
from casim.run import compose_scenario, prepare, run

RECORDED_REPLAY = Path(__file__).resolve().parents[1] / "scenarios" / "recorded_replay"
COMPONENTS = ("casim.pipelines.problem_based_template.InstanceLoader",
              "casim.pipelines.problem_based_template.OrdersProvider",
              "casim.pipelines.subproblems.item_assingment.RecordedIA",
              "casim.pipelines.subproblems.batching.RecordedTours",
              "casim.pipelines.subproblems.picker_routing.UShape")
FIFO = f"cosy_repo.components=[{','.join(COMPONENTS)},casim.pipelines.subproblems.scheduling.FIFOScheduler]"
PUSH = ("decisions.OBRSP.state_information.resources=[idle,reserved,off_shift]",)
PULL = ("decisions.OBRSP.state_information.resources=[idle]",
        "decisions.OBRSP.commitment_policy._target_="
        "casim.decision_engine.commitment_policies.PullCommitmentPolicy",
        "decisions.OBRSP.triggers=[OrderStreamClosed,PickerIdle,PickerArrival]",
        "decisions.OBRSP.scope.freeze_from=scheduled")


def _run(tmp_path, *overrides):
    cfg = compose_scenario(RECORDED_REPLAY, ["source.presence=recorded", *overrides],
                           output_dir=tmp_path)
    run(cfg)
    return json.loads((tmp_path / "service.json").read_text())


def test_push_strands_a_tour_with_a_picker_who_left_and_pull_does_not(tmp_path):
    push = _run(tmp_path / "push", FIFO, *PUSH)
    assert push["stranded"] == {"tours": 1, "orders": 1}
    assert push["orders"]["all"]["open_orders"] == 1
    pull = _run(tmp_path / "pull", FIFO, *PULL)
    assert pull["stranded"] == {"tours": 0, "orders": 0}
    assert pull["orders"]["all"]["open_orders"] == 0


def test_the_recorded_plan_cannot_be_pulled(tmp_path):
    cfg = compose_scenario(RECORDED_REPLAY, list(PULL), output_dir=tmp_path)
    with pytest.raises(ValueError, match="not supported by \\['RecordedPlan'\\]"):
        prepare(cfg)


PULL_DECISION = {
    "problem_class": "OBRSP",
    "state_information": {"orders": True, "tours": ["planned"], "resources": ["idle"]},
    "scope": {"freeze_from": "scheduled", "frozen_accepts": "none"},
    "solver": {"repo": {}},
    "commitment_policy": {"_target_": "casim.decision_engine.commitment_policies."
                                      "PullCommitmentPolicy"},
    "triggers": ["OrderStreamClosed", "PickerIdle", "PickerArrival"],
}


@pytest.mark.parametrize("change, message", [
    ({"state_information": {"orders": True, "tours": ["planned"],
                            "resources": ["idle", "reserved"]}}, "resources: \\[idle\\]"),
    ({"triggers": ["OrderStreamClosed", "PickerIdle"]}, "PickerArrival"),
    ({"scope": {"freeze_from": "planned", "frozen_accepts": "none"}}, "rebuildable"),
])
def test_a_pull_decision_must_see_only_free_pickers_and_be_called_by_them(change, message):
    DecisionConfig.model_validate(PULL_DECISION)
    with pytest.raises(ValidationError, match=message):
        DecisionConfig.model_validate({**PULL_DECISION, **change})


def test_each_free_picker_takes_one_tour():
    def create(picker, start):
        return CreateTour(SimpleNamespace(picker_id=picker, start_time=start))
    snapshot = SimpleNamespace(
        resources=SimpleNamespace(resources=[SimpleNamespace(id=0), SimpleNamespace(id=1)]),
        dynamic_warehouse_info=SimpleNamespace(plan=SimpleNamespace(tours=()),
                                               memory=SimpleNamespace(waiting=False)))
    commands = (create(0, 10.0), create(1, 10.0), create(0, 50.0), create(1, 60.0))
    commits, reconsider = PullCommitmentPolicy().apply(commands, snapshot)
    assert [commit.creation_index for commit in commits] == [0, 1]
    assert reconsider is None
