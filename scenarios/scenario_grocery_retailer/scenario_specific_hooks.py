import math

import numpy as np
from casim.events.operational_events import TruckDeparture, WMSRun, ShiftStart, PickerArrival, BreakStart, \
    TruckDisruption

DAY_SEC = 86400
SHIFT_START_SEC = 6 * 3600
WMS_RUN_SEC = 2 * 3600
BREAK_1_SEC = 8 * 3600
BREAK_2_SEC = 11 * 3600


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

# def make_shift_end_hook(n_days: int, shift_end_sec: float = 22 * 3600):
#     def hook(sim, domain):
#         for day in range(n_days):
#             sim.add_event(ShiftEnd(time=day * DAY_SEC + shift_end_sec))
#     return hook

def wms_run_hook(sim, domain):
    sim.add_event(WMSRun(time=2 * 3600))

def break_start_hook(sim, domain):
    sim.add_event(BreakStart(time=43200, break_duration=1800))
    sim.add_event(BreakStart(time=32400, break_duration=1800))

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

def make_picker_arrival_hook(n_days: int):
    daily_pickers = [25, 17, 19, 20, 18, 7]
    def hook(sim, domain):
        for day in range(n_days):
            t = day * DAY_SEC + SHIFT_START_SEC
            for resource in domain.resources.resources:
                sim.add_event(PickerArrival(time=t, picker_id=resource.id))
    return hook


def make_shift_start_hook(n_days: int):
    def hook(sim, domain):
        for day in range(n_days):
            sim.add_event(ShiftStart(time=day * DAY_SEC + SHIFT_START_SEC))
    return hook


def make_wms_run_hook(n_days: int):
    def hook(sim, domain):
        for day in range(n_days):
            sim.add_event(WMSRun(time=day * DAY_SEC + WMS_RUN_SEC))
    return hook


def make_break_hook(n_days: int):
    def hook(sim, domain):
        for day in range(n_days):
            base = day * DAY_SEC
            sim.add_event(BreakStart(time=base + BREAK_1_SEC, break_duration=1800))
            sim.add_event(BreakStart(time=base + BREAK_2_SEC, break_duration=1800))
    return hook

def make_truck_disruption_hook(n_days: int):
    def hook(sim, domain) -> None:
        sim.add_event(TruckDisruption(time=SHIFT_START_SEC + (3600 * 24), from_time=133200, to_time=122400, te_volume=100, palett_te_factor=1.6))

    return hook
