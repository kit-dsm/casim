from itertools import product

import pytest

from casim.pipelines.taxonomy import TAXONOMY, variant


def test_all_online_variants_are_ordered_or_reject_by_name():
    assert not {"ATIP", "RORSP", "OBRSPW"} & TAXONOMY.keys()
    assert TAXONOMY["OBRSP"]["variables"] == [
        "item_assignment", "batching", "routing", "scheduling",
        "sequencing", "waiting", "positioning", "admission",
        "active_tour_routing",
    ]
    for problem_class, incumbent, origin in product(
            TAXONOMY, (False, True), ("depot", "live")):
        features = {"incumbent": incumbent, "routing_origin": origin}
        if origin == "live" and (incumbent or problem_class != "OBRSP"):
            with pytest.raises(ValueError, match="O2|live-origin"):
                variant(problem_class, features)
            continue
        stages = variant(problem_class, features)
        positions = [TAXONOMY[problem_class]["variables"].index(stage)
                     for stage in stages]
        assert positions == sorted(positions)
