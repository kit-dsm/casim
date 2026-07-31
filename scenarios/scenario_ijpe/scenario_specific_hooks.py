from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from ware_ops_algos.domain_models import OrderPosition, Order

from casim.events.operational_events import (
    BreakStart,
    OrderArrival,
    PickerArrival,
    ShiftStart,
    TruckDeparture,
    TruckDisruption,
    WMSRun, VolumeShiftAcrossDay, OrderIngestion,
)
from scenarios.scenario_ijpe.generator.public.generate_order import generate_n_orders
from scenarios.scenario_ijpe.schema import COL_ORDER_ID, COL_ARTICLE_ID, COL_QUANTITY, \
    COL_ORDER_DATE, COL_DUE_DATE

DAY_SEC = 86400
HOUR_SEC = 3600


def h(hour: float) -> int:
    return int(hour * HOUR_SEC)


def t(day: int, hour: float, day_sec: int = DAY_SEC) -> int:
    return int(day * day_sec + hour * HOUR_SEC)


def add_orders_hook(sim, domain) -> None:
    for order in domain.orders.orders:
        sim.add_order(order)


def make_picker_arrival_hook(n_days: int, arrival_hour: float, day_sec: int, n_pickers_per_day: list[int]):
    assert n_days == len(n_pickers_per_day)

    def hook(sim, domain) -> None:
        # Deterministic max resource pool
        resources = sorted(domain.resources.resources, key=lambda r: r.id)
        max_pickers = len(resources)

        for day in range(n_days):
            arrival_time = t(day, arrival_hour, day_sec)
            n_available = n_pickers_per_day[day]

            assert n_available <= max_pickers, print(n_available, max_pickers)

            for idx, resource in enumerate(resources):
                sim.add_event(
                    PickerArrival(
                        time=arrival_time,
                        picker_id=resource.id,
                        picker_available=idx < n_available,
                    )
                )

    return hook


def make_shift_start_hook(n_days: int, shift_start_hour: float, day_sec: int):
    def hook(sim, domain) -> None:
        for day in range(n_days):
            sim.add_event(
                ShiftStart(
                    time=t(day, shift_start_hour, day_sec)
                )
            )

    return hook


def make_wms_run_hook(n_days: int, wms_run_hour: float, day_sec: int):
    def hook(sim, domain) -> None:
        for day in range(n_days):
            sim.add_event(
                WMSRun(
                    time=t(day, wms_run_hour, day_sec)
                )
            )

    return hook


def make_break_hook(n_days: int, breaks, day_sec: int):
    def hook(sim, domain) -> None:
        for day in range(n_days):
            base = day * day_sec

            for break_cfg in breaks:
                sim.add_event(
                    BreakStart(
                        time=base + h(break_cfg.start_hour),
                        break_duration=int(break_cfg.duration_minutes * 60),
                    )
                )

    return hook


def make_dock_manager_hook(K_dock: int):
    def hook(sim, domain) -> None:
        sim.state.configure_dock(K_dock)

    return hook


def make_truck_schedule_hook(
    n_days: int,
    start_hour: float,
    end_hour: float,
    interval_hours: float,
    capacity: int | None,
    day_sec: int,
):
    def hook(sim, domain) -> None:
        start_sec = h(start_hour)
        end_sec = h(end_hour)
        interval_sec = h(interval_hours)

        for day in range(n_days):
            base = day * day_sec
            truck_time = start_sec

            while truck_time <= end_sec:
                sim.add_event(
                    TruckDeparture(
                        time=base + truck_time,
                        capacity=capacity,
                    )
                )
                truck_time += interval_sec

    return hook


def make_truck_disruption_hook(
    disruption_day: int,
    disruption_hour: float,
    from_day: int,
    from_hour: float,
    to_day: int,
    to_hour: float,
    te_volume: int,
    day_sec: int,
    palett_te_factor: float = 1.6,
):
    disruption_time = t(disruption_day, disruption_hour, day_sec)
    from_time = t(from_day, from_hour, day_sec)
    to_time = t(to_day, to_hour, day_sec)

    def hook(sim, domain) -> None:
        sim.add_event(
            TruckDisruption(
                time=disruption_time,
                from_time=from_time,
                to_time=to_time,
                te_volume=te_volume,
                palett_te_factor=palett_te_factor,
            )
        )

    return hook


def make_cross_day_volume_shift_hook(
    disruption_day: int,
    disruption_hour: float,
    from_day: int,
    from_hour: float,
    to_day: int,
    to_hour: float,
    te_volume: int,
    day_sec: int,
    palett_te_factor: float = 1.6,
):
    disruption_time = t(disruption_day, disruption_hour, day_sec)
    from_time = t(from_day, from_hour, day_sec)
    to_time = t(to_day, to_hour, day_sec)

    def hook(sim, domain) -> None:
        sim.add_event(VolumeShiftAcrossDay(
            disruption_time,
            from_due=from_time,
            new_due=to_time,
            max_orders=max(1, int(te_volume / palett_te_factor)),
        ))

    return hook

def make_generated_order_injection_hook(
    calibration_path: Path,
    trigger_day: int,
    trigger_hour: float,
    due_day: int,
    due_hour: float,
    target_te: float,
    customer_id: str,
    order_prefix: str,
    seed: int,
    day_sec: int,
    order_builder,
    palett_te_factor: float = 1.6,
):
    trigger_time = t(trigger_day, trigger_hour, day_sec)
    due_time = t(due_day, due_hour, day_sec)

    calibration = json.loads(Path(calibration_path).read_text())

    def hook(sim, domain) -> None:
        n_orders = math.ceil(target_te / palett_te_factor)
        available_articles = {
            str(article.article_id) for article in domain.articles.articles
        }
        local_calibration = {
            **calibration,
            "articles": [
                article
                for article in calibration["articles"]
                if str(article["article_id"]) in available_articles
            ],
        }

        rows = generate_n_orders(
            calibration=local_calibration,
            n_orders=n_orders,
            order_date=trigger_time,
            due_date=due_time,
            customer_id=customer_id,
            order_prefix=order_prefix,
            seed=seed,
            day_sec=day_sec,
        )

        injected_orders = order_builder(rows, domain)

        for order in injected_orders:
            sim.add_order(order)

        sim.add_event(WMSRun(trigger_time))
        sim.add_event(OrderIngestion(trigger_time + 1e-6))
        print(
            f"Generated order injection scheduled {len(injected_orders)} orders "
            f"/ approx. {len(injected_orders) * palett_te_factor:.1f} TE "
            f"at {trigger_time}, due {due_time}."
        )

    return hook

def orders_from_canonical_rows(rows: pd.DataFrame, domain):
    order_list = []

    for o_id in rows[COL_ORDER_ID].unique():
        subset = rows[rows[COL_ORDER_ID] == o_id]

        positions = [
            OrderPosition(
                order_number=str(o_id),
                article_id=str(getattr(r, COL_ARTICLE_ID)),
                amount=int(getattr(r, COL_QUANTITY)),
            )
            for r in subset.itertuples(index=False)
        ]

        order_list.append(
            Order(
                order_id=str(o_id),
                order_date=float(subset[COL_ORDER_DATE].iloc[0]),
                due_date=float(subset[COL_DUE_DATE].iloc[0]),
                order_positions=positions,
            )
        )
    return order_list

def build_sim_hooks(cfg):
    sim_cfg = cfg.simulation

    n_days = int(sim_cfg.n_days)
    day_sec = int(sim_cfg.day_sec)
    pickers_per_day = list(sim_cfg.pickers_per_day)

    time_model = sim_cfg.time_model
    truck_schedule = time_model.truck_schedule

    hooks = [
        # Must be first! Cross day shifts rewrite these raw OrderArrival events.
        add_orders_hook,
    ]

    for injection in sim_cfg.hooks.generated_order_injections:
        hooks.append(
            make_generated_order_injection_hook(
                calibration_path=Path(injection.calibration_path),
                trigger_day=int(injection.trigger_day),
                trigger_hour=float(injection.trigger_hour),
                due_day=int(injection.due_day),
                due_hour=float(injection.due_hour),
                target_te=float(injection.target_te),
                customer_id=str(injection.customer_id),
                order_prefix=str(injection.order_prefix),
                seed=int(injection.seed),
                day_sec=day_sec,
                order_builder=orders_from_canonical_rows,
                palett_te_factor=float(injection.palett_te_factor),
            )
        )

    for shift in sim_cfg.hooks.cross_day_volume_shifts:
        hooks.append(
            make_cross_day_volume_shift_hook(
                disruption_day=int(shift.disruption_day),
                disruption_hour=float(shift.disruption_hour),
                from_day=int(shift.from_day),
                from_hour=float(shift.from_hour),
                to_day=int(shift.to_day),
                to_hour=float(shift.to_hour),
                te_volume=int(shift.te_volume),
                day_sec=day_sec,
            )
        )

    hooks.extend(
        [
            make_picker_arrival_hook(
                n_days=n_days,
                arrival_hour=float(time_model.picker_arrival_hour),
                day_sec=day_sec,
                n_pickers_per_day=pickers_per_day
            ),
            make_shift_start_hook(
                n_days=n_days,
                shift_start_hour=float(time_model.shift_start_hour),
                day_sec=day_sec,
            ),
            make_truck_schedule_hook(
                n_days=n_days,
                start_hour=float(truck_schedule.start_hour),
                end_hour=float(truck_schedule.end_hour),
                interval_hours=float(truck_schedule.interval_hours),
                capacity=truck_schedule.capacity,
                day_sec=day_sec,
            ),
            make_wms_run_hook(
                n_days=n_days,
                wms_run_hour=float(time_model.wms_run_hour),
                day_sec=day_sec,
            ),
            make_dock_manager_hook(
                K_dock=int(sim_cfg.dock.K_dock),
            ),
            make_break_hook(
                n_days=n_days,
                breaks=time_model.breaks,
                day_sec=day_sec,
            ),
        ]
    )

    for disruption in sim_cfg.hooks.truck_disruptions:
        hooks.append(
            make_truck_disruption_hook(
                disruption_day=int(disruption.disruption_day),
                disruption_hour=float(disruption.disruption_hour),
                from_day=int(disruption.from_day),
                from_hour=float(disruption.from_hour),
                to_day=int(disruption.to_day),
                to_hour=float(disruption.to_hour),
                te_volume=int(disruption.te_volume),
                day_sec=day_sec,
            )
        )

    return hooks
