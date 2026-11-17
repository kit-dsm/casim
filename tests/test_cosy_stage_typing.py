"""CoSy synthesizes only the valid routing families (Task 4.13).

A repository with FiFo, AdmittedTourBatch, SShape and ReroutableSShape could, with one
untyped batch stage, pair FiFo with rerouting and admission with depot routing. Depot routing
takes AbstractBatching and active-tour routing takes AdmittedTourBatch, so exactly two
families remain: FiFo -> SShape and admission -> AdmittedTourBatch -> ReroutableSShape.
Algorithm cards are left out (no models), so the result shows the stage types only.
"""

from pathlib import Path

import yaml
from cosy.maestro import Maestro
from cosy_luigi import CoSyLuigiRepo

from casim.config.data_card import load_and_flatten_data_card
from casim.pipelines.problem_based_template import (
    ActiveTourRoutingPlanCandidate, AdmittedTourBatch, InstanceLoader, OrdersProvider,
    PlanCandidateEndpoint, RoutingPlanCandidate, traverse_pipeline,
)
from casim.pipelines.subproblems.admission import RemainingRouteAdmissionNode
from casim.pipelines.subproblems.batching import FiFo
from casim.pipelines.subproblems.item_assingment import GreedyIA
from casim.pipelines.subproblems.picker_routing import ReroutableSShape, SShape

DATA_CARD = (Path(__file__).resolve().parents[1] / "scenarios" / "analytic_waiting"
             / "data_card" / "stochastic_waiting.yaml")


def test_repository_synthesizes_exactly_the_two_valid_families():
    data_card = load_and_flatten_data_card(yaml.safe_load(DATA_CARD.read_text(encoding="utf-8")))
    PlanCandidateEndpoint.configure(data_card, [])
    classes = (InstanceLoader, OrdersProvider, GreedyIA, FiFo, SShape,
               RemainingRouteAdmissionNode, AdmittedTourBatch, ReroutableSShape,
               RoutingPlanCandidate, ActiveTourRoutingPlanCandidate, PlanCandidateEndpoint)
    repo = CoSyLuigiRepo(*classes)
    pipelines = list(Maestro(repo.cls_repo, repo.taxonomy).query(PlanCandidateEndpoint.target()))
    families = sorted(
        tuple(sorted(cls.__name__ for cls in {type(task) for task in traverse_pipeline([pipeline])}
                     if cls in (FiFo, SShape, AdmittedTourBatch, ReroutableSShape)))
        for pipeline in pipelines)
    assert families == [("AdmittedTourBatch", "ReroutableSShape"), ("FiFo", "SShape")]
