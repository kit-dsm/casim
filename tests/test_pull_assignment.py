"""Push and pull assignment (docs/concepts.md, Resources and dispatch).

Push commits tours into picker queues ahead of time; pull lets a free picker
cause the decision (``PickerIdle``), which commits one tour. In recorded_replay
with recorded presence, picker 1 is present from 35 s to 50 s and picker 0 is away from 30 s to about 103 s. A
FIFO push at the start queues order 4's tour with picker 1, who leaves before
it can start it; with pull, picker 0 takes it when it is back.
"""

import json
from pathlib import Path

import pytest

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
        "casim.decision_engine.commitment_policies.SchedulingCommitmentPolicy",
        "+decisions.OBRSP.commitment_policy.n_jobs=1",
        "decisions.OBRSP.triggers=[OrderStreamClosed,PickerIdle,PickerArrival]",
        "decisions.OBRSP.scope.freeze_from=scheduled",
        "decisions.OBRSP.conditions=[{_target_:"
        "casim.simulation_engine.conditions.NbrPickersCondition,threshold:1}]")


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
    # Four one-order tours, and each picker has time for two: one order stays
    # open with pull too, but uncommitted, not stranded with a picker who left.
    assert pull["orders"]["all"]["open_orders"] == 1


def test_the_recorded_plan_cannot_be_pulled(tmp_path):
    cfg = compose_scenario(RECORDED_REPLAY, list(PULL), output_dir=tmp_path)
    with pytest.raises(ValueError, match="\\['RecordedPlan'\\] cannot be pulled"):
        prepare(cfg)
