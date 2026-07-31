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
    "RORSP": {
            "variables": ["routing", "scheduling"],
            "endpoint": "casim.pipelines.problem_based_template.ResultAggregationScheduling"
        },
}
