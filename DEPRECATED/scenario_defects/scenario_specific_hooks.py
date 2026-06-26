import math
from dataclasses import dataclass

import numpy as np
from ware_ops_algos.algorithms import PickList

from casim.events.operational_events import TruckDeparture, WMSRun, ShiftStart, PickerArrival

SHIFT_START_SEC = 7 * 3600


@dataclass
class Tour:
    order_date: int
    n_picks: int
    resource_id: int
    release: int

def picker_arrival_hook(sim,
                        domain):
    min_order_date = np.inf
    for o in domain.orders.orders:
        if o.order_date < min_order_date:
            min_order_date = o.order_date
    for resource in domain.resources.resources:
        sim.add_event(PickerArrival(time=SHIFT_START_SEC,
                                    picker_id=resource.id))


def add_tours_hook(sim,
                   domain,
                   seed):
    np.random.seed(seed)

    Tour(np.random.randint(0,100))
    orders = domain.orders.orders
    for order in orders:
        sim.add_order(order)


def shift_start_hook(sim, domain):
    sim.add_event(ShiftStart(time=SHIFT_START_SEC))

