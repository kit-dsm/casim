def solution_kind(problem_class: str) -> str:
    """Derive the solution kind from the taxonomy variables.

    A problem whose variables include ``scheduling`` returns a
    ``SchedulingSolution``; one that includes ``routing`` returns a
    ``CombinedRoutingSolution``; otherwise a ``BatchingSolution``.
    """
    variables = set(TAXONOMY[problem_class]["variables"])
    if "scheduling" in variables:
        return "SchedulingSolution"
    if "routing" in variables:
        return "CombinedRoutingSolution"
    return "BatchingSolution"


TAXONOMY = {
    "OBRP": {
        "variables": ["item_assignment", "batching", "routing"],
        "endpoint": "casim.pipelines.problem_based_template.ResultAggregationRouting"
    },
    "SPRP": {
        "variables": ["routing"]
    },
    "ORP": {
       "variables": ["routing"],
        "endpoint": "casim.pipelines.problem_based_template.ResultAggregationRouting"
    },
    "OBP": {
        "variables": ["item_assignment", "batching"],
        "endpoint": "casim.pipelines.problem_based_template.ResultAggregationBatching"
    },
    "OSBP": {
            "variables": ["order_splitting", "item_assignment", "batching"],
            "endpoint": "casim.pipelines.problem_based_template.ResultAggregationBatching"
        },
    "BSRP": {
       "variables": ["batching", "routing"]
    },
    "OBRSP": {
       "variables": ["item_assignment", "batching", "routing", "scheduling"],
        "endpoint": "casim.pipelines.problem_based_template.ResultAggregationScheduling"
    },
    "ORSP": {
        "variables": ["routing", "scheduling"],
        "endpoint": "casim.pipelines.problem_based_template.ResultAggregationScheduling"
    },
}
