SUBPROBLEMS = {
    # Offline + Single Picker + Simple
    "SPRP": {
        "objectives": ["distance", "picking_time"],
        "variables": ["item_assignment", "order_selection", "routing"]
    },
    "SPRP-SS": {
            "objectives": ["distance", "picking_time"],
            "variables": ["item_assignment", "order_selection", "routing"]
        },
    "OBP": {
        "objectives": ["tardiness", "picking_time", "cost"],
        "variables": ["item_assignment", "order_selection", "batching"]
    },
    "OBSP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["item_assignment", "order_selection", "batching", "sequencing"]
    },
    "OBRP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["item_assignment", "batching", "routing", "batching_routing"]
    },
    "ORP": {
            "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
            "variables": ["routing"]
        },
    "OBSRP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["item_assignment", "batching", "sequencing", "routing"]
    },
    "OSRP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["item_assignment", "order_selection", "sequencing", "routing"]
    },
    "OBASRP": {
            "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
            "variables": ["item_assignment", "order_selection", "assignment", "batching", "sequencing", "routing"]
        },

    "OBARP": {
                "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
                "variables": ["item_assignment", "order_selection", "assignment", "batching", "sequencing", "routing"]
            },

    # Offline + Single Picker + Joint
    "OBPMP": {
        "objectives": ["tardiness", "picking_time", "cost"],
        "variables": ["batching"]
    },
    "OBSPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["batching", "sequencing"]
    },
    "OBAPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["batching", "assigning"]
    },
    "OBRPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
        "variables": ["batching", "routing"]
    },
    "OBSAPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time"],
        "variables": ["batching", "sequencing", "assigning"]
    },
    "OBSRPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time"],
        "variables": ["batching", "sequencing", "routing"]
    },
    "OBSARPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time"],
        "variables": ["batching", "sequencing", "assigning", "routing"]
    },

    # Offline + Multiple Pickers + Simple
    "OOBP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching"]
    },
    "OOBSP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "sequencing"]
    },
    "OOBAP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "assigning"]
    },
    "OOBWP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "waiting"]
    },
    "OOBSRP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "sequencing", "routing"]
    },
    "OOBSWP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "sequencing", "waiting"]
    },
    "OOBRWP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "routing", "waiting"]
    },
    "OOBSRWP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "sequencing", "routing", "waiting"]
    },

    # Offline + Multiple Pickers + Joint
    "OOBPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching"]
    },
    "OOBSPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "sequencing"]
    },
    "OOBAPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "assigning"]
    },
    "OOBWPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "waiting"]
    },
    "OOBSAPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "sequencing", "assigning"]
    },
    "OOBSRPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "sequencing", "routing"]
    },
    "OOBSWPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "sequencing", "waiting"]
    },
    "OOBRWPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "routing", "waiting"]
    },
    "OOBARPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "assigning", "routing"]
    },
    "OOBSARPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time"],
        "variables": ["batching", "sequencing", "assigning", "routing"]
    },
    "OOBAWPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "assigning", "waiting"]
    },
    "OOBSARWPMP": {
        "objectives": ["tardiness", "picking_time", "cost", "completion_time", "turnaround_time", "blocking_time",
                       "workload_balance"],
        "variables": ["batching", "sequencing", "assigning", "routing", "waiting"]
    }
}