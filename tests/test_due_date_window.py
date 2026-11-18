"""A due-date window needs due dates and applies after the stream's end (Task 4.15)."""

from pathlib import Path

import pytest
from omegaconf import open_dict

from casim.run import check
from casim.config.scenario_check import compose_scenario

ROOT = Path(__file__).resolve().parents[1]


def test_window_by_due_date_needs_due_date_in_the_data_card():
    cfg = compose_scenario(ROOT / "scenarios" / "henn_waiting",
                           ["decisions=release_only_example"])
    with open_dict(cfg):
        cfg.decisions.planning.state_information.window = {"by": "due_date",
                                                           "horizon_s": 3600}
    with pytest.raises(ValueError, match="due_date"):
        check(cfg)
