TAXONOMY = {
    "OBRP": {
        "variables": ["item_assignment", "batching", "routing"],
        "endpoint": "ResultAggregationRouting"
    },
    "SPRP": {
        "variables": ["routing"]
    },
    "ORP": {
       "variables": ["routing"],
        "endpoint": "ResultAggregationRouting"
    },
    "OBP": {
        "variables": ["item_assignment", "batching"],
        "endpoint": "ResultAggregationBatching"
    },
    "BSRP": {
       "variables": ["batching", "routing"]
    },
    "OBRSP": {
       "variables": ["item_assignment", "batching", "routing", "sequencing"],
        "endpoint": "ResultAggregationSequencing"
    },
    "ORSP": {
        "variables": ["routing", "scheduling"],
        "endpoint": "ResultAggregationScheduling"
    },
}
