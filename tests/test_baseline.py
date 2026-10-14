"""The recorded Phase B configurations remain unchanged unless re-recorded."""

import pytest

from scripts.baseline import compare


@pytest.mark.baseline
def test_recorded_scenario_baselines():
    assert compare() == []
