"""Translate a decision-visible started tour into admission algorithm input."""

from ware_ops_algos.algorithms import AdmissionTour
from ware_ops_algos.planning.models import TourView


def to_admission_tour(view: TourView) -> AdmissionTour:
    if (view.picker_id is None or view.pick_cart is None
            or view.occupied_bins is None or view.routing_origin is None):
        raise ValueError(f"Tour {view.tour_id} lacks admission execution facts")
    return AdmissionTour(
        tour_id=view.tour_id,
        picker_id=view.picker_id,
        pick_cart=view.pick_cart,
        active_order_ids=view.active_order_ids,
        remaining_route=view.remaining_route,
        occupied_bins=view.occupied_bins,
        routing_origin=view.routing_origin,
        picking_until=view.picking_until,
    )
