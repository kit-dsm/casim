from casim.events.operational_events import Event
from casim.events.operational_events import (
    BreakStart,
    FlushRemainingOrders,
    PickerArrival,
    PickerDeparture,
    PlanningRun,
    WMSRun,
)

DAY = 24 * 60 * 60


class DueTimeDisruption(Event):
    """One concrete study disruption; state owns the actual transition."""

    def __init__(self, time, from_due, new_due, max_orders):
        super().__init__(time)
        self.from_due = from_due
        self.new_due = new_due
        self.max_orders = max_orders

    def handle(self, state):
        state.disrupt_unstarted_work(
            from_due=self.from_due,
            new_due=self.new_due,
            max_orders=self.max_orders,
        )
        return [PlanningRun(self.time)]


def build_hooks(cfg):
    def add_study_events(simulation, domain):
        spec = domain.dynamic_operations_spec
        orders = sorted(
            domain.orders.orders,
            key=lambda value: (value.order_date, value.order_id),
        )
        for order in orders:
            simulation.add_order(order)

        picker_counts = [int(value) for value in spec["picker_counts"]]
        shift_start = float(spec.get("shift_start_hour", 6)) * 3600
        shift_end = float(spec.get("shift_end_hour", 18)) * 3600
        for day, picker_count in enumerate(picker_counts):
            day_start = day * DAY
            for picker in domain.resources.resources:
                simulation.add_event(
                    PickerArrival(
                        day_start + shift_start,
                        picker.id,
                        picker.id < picker_count,
                    )
                )
                simulation.add_event(
                    PickerDeparture(day_start + shift_end, picker.id)
                )
            for break_spec in spec.get("breaks", []):
                simulation.add_event(
                    BreakStart(
                        day_start
                        + float(break_spec["start_hour"]) * 3600,
                        float(break_spec["duration_minutes"]) * 60,
                    )
                )
            for hour in cfg.engines.wms_hours:
                simulation.add_event(WMSRun(day_start + float(hour) * 3600))
            planning_hours = list(cfg.engines.planning_hours)
            interval = cfg.engines.get("planning_interval_minutes")
            if interval is not None:
                minute = int(float(spec.get("shift_start_hour", 6)) * 60)
                end_minute = int(float(spec.get("shift_end_hour", 18)) * 60)
                while minute < end_minute:
                    planning_hours.append(minute / 60)
                    minute += int(interval)
            for hour in sorted(set(planning_hours)):
                simulation.add_event(
                    PlanningRun(day_start + float(hour) * 3600)
                )

        if bool(cfg.engines.get("disruption_replanning", False)):
            disruption = spec["due_time_disruption"]
            day = int(disruption["day"])
            simulation.add_event(
                DueTimeDisruption(
                    day * DAY
                    + float(disruption["event_hour"]) * 3600,
                    day * DAY
                    + float(disruption["from_due_hour"]) * 3600,
                    day * DAY
                    + float(disruption["new_due_hour"]) * 3600,
                    int(disruption["max_orders"]),
                )
            )
        close_time = max(
            (float(order.order_date or 0) for order in orders),
            default=0.0,
        )
        simulation.add_event(FlushRemainingOrders(close_time))

    return [add_study_events]
