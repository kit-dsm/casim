"""The L1 pipelines: CASOP v0.1.0 labels, current ware_ops_algos classes, casim components."""

from __future__ import annotations

from dataclasses import dataclass
import re

TEMPLATE = "casim.pipelines.problem_based_template"
SUBPROBLEMS = "casim.pipelines.subproblems"


@dataclass(frozen=True)
class Pipeline:
    """One pipeline of Task B.4, named by its label in CASOP's v0.1.0 results."""

    label: str
    item_assignment: str
    batching: str
    routing: str
    scheduling: str = "FIFOScheduler"
    batching_router: str | None = None
    batching_start: str | None = None

    @property
    def batching_component(self) -> str:
        if self.batching in ("FiFo", "DueDate"):
            return f"{SUBPROBLEMS}.batching.{self.batching}"
        return f"{SUBPROBLEMS}.generated.{_module(self.batching)}.{self.batching}"

    def components(self) -> list[str]:
        """The CoSy repository of the CASIM decision."""
        return [
            f"{TEMPLATE}.InstanceLoader",
            f"{TEMPLATE}.OrdersProvider",
            f"{SUBPROBLEMS}.item_assingment.{self.item_assignment}",
            self.batching_component,
            f"{SUBPROBLEMS}.picker_routing.{self.routing}",
            f"{SUBPROBLEMS}.scheduling.{self.scheduling}",
            f"{SUBPROBLEMS}.waiting.NoWaitingNode",
        ]


def _module(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# v0.1.0 label -> pipeline. "SingleOrderBatching" was a ware_ops_pipes template
# stage, not an algorithm card; every SPRP and SPRP-SS instance has one order,
# so FiFo with one box for the whole order returns the same single batch.
PIPELINES = {p.label: p for p in (
    Pipeline("GIA+SingleOrderBatching+RR", "GreedyIA", "FiFo", "RatliffRosenthal"),
    Pipeline("GIA+SingleOrderBatching+SShape", "GreedyIA", "FiFo", "SShape"),
    Pipeline("MinMinIA+SingleOrderBatching+RR", "MinMinIA", "FiFo", "RatliffRosenthal"),
    Pipeline("MinMinIA+SingleOrderBatching+SShape", "MinMinIA", "FiFo", "SShape"),
    Pipeline("GIA+LSBatchingRROrderNrFiFo+RR", "GreedyIA", "LSBatchingRROrderNrFiFo",
             "RatliffRosenthal", batching_router="RatliffRosenthal", batching_start="OrderNrFiFo"),
    Pipeline("GIA+LSBatchingSShapeFiFoOrderNr+SShape", "GreedyIA", "LSBatchingSShapeFiFoOrderNr",
             "SShape", batching_router="SShape", batching_start="OrderNrFiFo"),
    Pipeline("GIA+SavingsSShape+SShape", "GreedyIA", "ClarkAndWrightSShape", "SShape",
             batching_router="SShape"),
    Pipeline("GIA+LSOrdNrNN+NN", "GreedyIA", "LSBatchingNNFiFoOrderNr", "NearestNeighbourhood",
             batching_router="NearestNeighbourhood", batching_start="OrderNrFiFo"),
    Pipeline("GIA+DueDate+NN+EDD", "GreedyIA", "DueDate", "NearestNeighbourhood",
             scheduling="EDDScheduler"),
)}
