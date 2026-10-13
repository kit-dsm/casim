"""Online problem classes and the subproblems synthesized for a decision."""

TAXONOMY = {
    "OBRP": {"variables": ["item_assignment", "batching", "routing"],
             "endpoint": "ResultAggregationRouting"},
    "SPRP": {"variables": ["routing"]},
    "ORP": {"variables": ["routing"],
            "endpoint": "ResultAggregationRouting"},
    "OBP": {"variables": ["item_assignment", "batching"],
            "endpoint": "ResultAggregationBatching"},
    "OSBP": {"variables": ["order_splitting", "item_assignment", "batching"],
             "endpoint": "ResultAggregationBatching"},
    "BSRP": {"variables": ["batching", "routing"]},
    "OBRSP": {"variables": [
        "item_assignment", "batching", "routing", "scheduling",
        "sequencing", "waiting", "positioning", "admission",
        "active_tour_routing",
    ], "endpoint": "ResultAggregationWaiting"},
    "ORSP": {"variables": ["routing", "scheduling"],
             "endpoint": "ResultAggregationScheduling"},
}


def variant(problem_class: str, features: dict[str, bool | str]) -> tuple[str, ...]:
    """Return the decision's ordered subproblem sequence."""
    if problem_class not in TAXONOMY:
        raise ValueError(f"Unknown online problem class {problem_class}")
    incumbent = features["incumbent"]
    origin = features["routing_origin"]
    if origin == "live":
        if incumbent:
            raise ValueError("Open point O2: cannot rebuild planned tours and admit to a live tour together")
        stages = ("item_assignment", "admission", "active_tour_routing")
        available = TAXONOMY[problem_class]["variables"]
        if not all(stage in available for stage in stages):
            raise ValueError(f"{problem_class} has no live-origin admission variant")
        return stages
    if origin != "depot":
        raise ValueError(f"Unknown routing_origin {origin}")
    return tuple(stage for stage in TAXONOMY[problem_class]["variables"]
                 if stage not in ("admission", "active_tour_routing"))
