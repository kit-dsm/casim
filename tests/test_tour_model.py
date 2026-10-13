from casim.domain_objects.tour_model import TourPlanningState
from ware_ops_algos.algorithms import BatchObject, Route
from ware_ops_algos.planning.lifecycle import TourStates


def test_tour_state_works_after_package_import():
    batch = BatchObject(batch_id=1, orders=[])
    route = Route(distance=0.0, batch=batch, annotated_route=[])
    tour = TourPlanningState(
        tour_id=1,
        order_numbers=[],
        original_route=route,
        route=route,
        batch=batch,
        annotated_route=[],
    )

    assert tour.status == TourStates.PLANNED
