import math

import numpy as np
from casim.events.operational_events import TruckDeparture, WMSRun, ShiftStart, PickerArrival

SHIFT_START_SEC = 7 * 3600


class DockManager:
    def __init__(self, K_dock: int):
        self.K_dock = K_dock
        self.n_staged_pallets: int = 0

    def stage_pallets(self, n_pallets: int = 1) -> None:
        self.n_staged_pallets += n_pallets

    def release_pallets(self, n_pallets: int = 1) -> None:
        self.n_staged_pallets = max(0, self.n_staged_pallets - n_pallets)


def picker_arrival_hook(sim,
                        domain):
    min_order_date = np.inf
    for o in domain.orders.orders:
        if o.order_date < min_order_date:
            min_order_date = o.order_date
    for resource in domain.resources.resources:
        sim.add_event(PickerArrival(time=SHIFT_START_SEC,
                                    picker_id=resource.id))


def add_orders_hook(sim,
                    domain):
    orders = domain.orders.orders
    for order in orders:
        sim.add_order(order)


def shift_start_hook(sim, domain):
    sim.add_event(ShiftStart(time=SHIFT_START_SEC))


def wms_run_hook(sim, domain):
    sim.add_event(WMSRun(time=2 * 3600))


def make_dock_manager_hook(K_dock: int = 98):
    def hook(sim, domain) -> None:
        sim.state.dock_manager = DockManager(K_dock=K_dock)
    return hook


def make_truck_schedule_hook(
    bin_minutes: int = 30,
    sweep_time_sec: float = 18 * 3600,
):
    def hook(sim, domain) -> None:
        orders_domain = domain.orders
        bin_sec = bin_minutes * 60

        bins = {}
        unmatched = 0

        for order in orders_domain.orders or []:
            if order.due_date is None:
                unmatched += 1
                continue

            edge = math.ceil(float(order.due_date) / bin_sec) * bin_sec
            bins[edge] = bins.get(edge, 0) + 1

        for edge, capacity in sorted(bins.items()):
            sim.add_event(TruckDeparture(time=edge, capacity=capacity))

        if unmatched:
            sim.add_event(
                TruckDeparture(time=sweep_time_sec, capacity=unmatched)
            )

    return hook