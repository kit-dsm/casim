"""Smoke tests for the CoSy components no scenario or benchmark runs (Task 4.16).

Each component is swapped into the pipeline of ``scenarios/casop_static`` on one
small static instance (or of ``scenarios/henn_waiting`` where a card needs one
picker and pick times); the run must complete every order. The test proves
the component synthesises, solves and its tours execute, not that its result
is good.
"""

import csv
import json
from pathlib import Path

import pytest

from casim.run import compose_scenario, run

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "casop_static"
P = "casim.pipelines"
LOADERS = (f"{P}.problem_based_template.InstanceLoader",
           f"{P}.problem_based_template.OrdersProvider",
           f"{P}.subproblems.item_assingment.GreedyIA")
DEFAULT = {"batching": f"{P}.subproblems.batching.FiFo",
           "routing": f"{P}.subproblems.picker_routing.SShape",
           "scheduling": f"{P}.subproblems.scheduling.FIFOScheduler",
           "waiting": f"{P}.subproblems.waiting.NoWaitingNode"}
HENN_WAESCHER = ()  # the scenario's default instance, 21s-20-30-1
KRIS = ("data_card=kris", "source.instance_set=KrisSmallDataCorrected",
        "source.instance='instances_9_1'")
HENN_ONLINE = ("scenario=henn_waiting",)
GENERATED = f"{P}.subproblems.generated"

CASES = {
    "OrderNrFiFo": ("batching", f"{P}.subproblems.batching.OrderNrFiFo", HENN_WAESCHER),
    "ClarkAndWrightNN": ("batching", f"{GENERATED}.clark_and_wright_n_n.ClarkAndWrightNN", HENN_WAESCHER),
    "ClarkAndWrightRR": ("batching", f"{GENERATED}.clark_and_wright_r_r.ClarkAndWrightRR", HENN_WAESCHER),
    "LSBatchingNNFiFo": ("batching", f"{GENERATED}.l_s_batching_n_n_fi_fo.LSBatchingNNFiFo", HENN_WAESCHER),
    "LSBatchingRRFiFo": ("batching", f"{GENERATED}.l_s_batching_r_r_fi_fo.LSBatchingRRFiFo", HENN_WAESCHER),
    "LSBatchingSShapeFiFo": ("batching", f"{GENERATED}.l_s_batching_s_shape_fi_fo.LSBatchingSShapeFiFo", HENN_WAESCHER),
    "LSBatchingNNDueDate": ("batching", f"{GENERATED}.l_s_batching_n_n_due_date.LSBatchingNNDueDate", KRIS),
    "LSBatchingSShapeDueDate": ("batching", f"{GENERATED}.l_s_batching_s_shape_due_date.LSBatchingSShapeDueDate", KRIS),
    "Midpoint": ("routing", f"{P}.subproblems.picker_routing.Midpoint", HENN_WAESCHER),
    "TSPRouting": ("routing", f"{P}.subproblems.picker_routing.TSPRouting", HENN_WAESCHER),
    "ERDScheduler": ("scheduling", f"{P}.subproblems.scheduling.ERDScheduler", HENN_WAESCHER),
    # needs resources.time_per_pick and exactly one picker: the online Henn scenario
    "SAVSchedulerLargestGap": ("scheduling", f"{GENERATED}.s_a_v_scheduler_largest_gap.SAVSchedulerLargestGap", HENN_ONLINE),
    # its card (LPTScheduler) needs resources.time_per_pick and orders.due_date
    "HennLongScheduler": ("scheduling", f"{P}.subproblems.scheduling.HennLongScheduler", KRIS),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_component_runs_a_static_instance(name, tmp_path):
    slot, component, overrides = CASES[name]
    scenario = SCENARIO
    if overrides and overrides[0].startswith("scenario="):
        scenario, overrides = ROOT / "scenarios" / overrides[0].split("=", 1)[1], overrides[1:]
    components = [*LOADERS, *{**DEFAULT, slot: component}.values()]
    cfg = compose_scenario(scenario, [*overrides, f"cosy_repo.components=[{','.join(components)}]"],
                           tmp_path)
    run(cfg)
    kpis = json.loads((tmp_path / "kpis.json").read_text(encoding="utf-8"))
    with open(tmp_path / "orders.csv", newline="", encoding="utf-8") as handle:
        orders = list(csv.DictReader(handle))
    assert kpis["num_tours"] > 0
    assert kpis["num_orders_completed"] == len(orders) > 0
