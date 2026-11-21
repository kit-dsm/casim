"""Resource states are logged only after events that can change them.

Travel, node arrivals and pick ends move a picker along its tour but change
neither its tour's status, its presence nor its queue, so the engine does not
log resource states after them. Logging after every event must give the same
idle, busy and absent time.
"""

import json

import pytest

from casim.events.operational_events import NodeArrival, PickComplete, TravelEvent
from casim.run import compose_scenario, run
from tests.test_pull_assignment import FIFO, PULL, PUSH, RECORDED_REPLAY

MOVEMENT = (TravelEvent, NodeArrival, PickComplete)


def _outputs(tmp_path, overrides):
    cfg = compose_scenario(RECORDED_REPLAY, overrides, output_dir=tmp_path)
    result = run(cfg)
    return (result.trace_digest,
            json.loads((tmp_path / "tracker.json").read_text()),
            json.loads((tmp_path / "kpis.json").read_text()))


@pytest.mark.parametrize("overrides", [
    ["source.presence=recorded", FIFO, *PULL],
    ["source.presence=recorded", FIFO, *PUSH],
    [],
])
def test_skipping_movement_events_keeps_the_resource_log(tmp_path, monkeypatch, overrides):
    assert not any(event.changes_resources for event in MOVEMENT)
    skipped = _outputs(tmp_path / "skipped", overrides)
    for event in MOVEMENT:
        monkeypatch.setattr(event, "changes_resources", True)
    every = _outputs(tmp_path / "every", overrides)
    assert skipped == every
