"""Apply waiting and release rules to every scheduled candidate."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ware_ops_algos.algorithms import (
    AnalyticStochasticWaiting, HennWaiting, NoWaiting, OrderCountWaiting,
    WaitingInput,
)
from ware_ops_algos.algorithms.routing.routing import SShapeRouting

from casim.config.schema import CommitmentConfig, ReleaseGates, WaitingConfig


@dataclass(frozen=True)
class ReleaseDecision:
    planned_jobs: tuple
    jobs: tuple
    reconsiderations: tuple[tuple[float, str], ...] = ()


def _single_order_service_times(snapshot, job, picker) -> dict[int, float]:
    layout = snapshot.layout.layout_network
    router = SShapeRouting(
        start_node=layout.start_node, end_node=layout.end_node,
        closest_node_to_start=layout.closest_node_to_start,
        min_aisle_position=layout.min_aisle_position,
        max_aisle_position=layout.max_aisle_position,
        distance_matrix=layout.distance_matrix,
        predecessor_matrix=layout.predecessor_matrix,
        picker=[picker], gen_tour=True, gen_item_sequence=True,
        node_list=layout.node_list,
        node_to_idx={node: index for index, node in
                     enumerate(layout.graph.nodes)},
        idx_to_node={index: node for index, node in
                     enumerate(layout.graph.nodes)},
    )
    service = {}
    for order in job.job.route.batch.orders:
        router.reset_parameters()
        route = router.solve(order.pick_positions).route
        service[order.order_id] = (
            picker.tour_setup_time + route.distance / picker.speed
            + sum(pick.in_store for pick in order.pick_positions)
            * picker.time_per_pick
        )
    return service


def release_decision(jobs, snapshot, waiting: WaitingConfig,
                     commitment: CommitmentConfig,
                     gates: ReleaseGates | None = None) -> ReleaseDecision:
    jobs = tuple(jobs)
    now = snapshot.context.time
    gates = gates or ReleaseGates()
    if (gates.outside_break and snapshot.operations.is_break or
            gates.max_staged_pallets is not None and
            snapshot.operations.n_staged_pallets > gates.max_staged_pallets):
        return ReleaseDecision(jobs, ())
    by_picker = {resource.id: resource for resource in snapshot.plan.resources}
    selected = []
    reconsider: list[tuple[float, str]] = []
    planned = []
    earliest_by_picker = {
        picker_id: min((index for index, job in enumerate(jobs)
                        if job.picker_id == picker_id),
                       key=lambda index: (jobs[index].start_time, index))
        for picker_id in {job.picker_id for job in jobs}
    }
    for index, job in enumerate(jobs):
        picker = by_picker[job.picker_id]
        policy = waiting.policy
        if policy != "no_waiting" and index != earliest_by_picker[job.picker_id]:
            planned.append(job)
            continue
        if (job.order_numbers and len(job.order_numbers) < waiting.min_orders
                and not snapshot.context.input_closed):
            planned.append(job)
            continue
        if policy == "no_waiting":
            waiter = NoWaiting()
        elif policy == "order_count":
            waiter = OrderCountWaiting(waiting.k)
        elif policy == "henn":
            waiter = HennWaiting()
        elif policy == "analytic_stochastic":
            waiter = AnalyticStochasticWaiting(
                target_batch_size_orders=waiting.target_batch_size_orders)
        else:
            raise ValueError(f"Unknown waiting policy {policy}")
        if policy == "henn" and len(jobs) > 1 and index == earliest_by_picker[job.picker_id]:
            planned.append(job)
            selected.append(job)
            continue
        decision = waiter.solve(WaitingInput(
            candidates=(job,), current_time=now,
            input_closed=snapshot.context.input_closed,
            picker=picker, layout=snapshot.layout,
            information=snapshot.information,
            deadline_reached="Reconsider:wait" in snapshot.context.cause,
            single_order_service_times=(
                _single_order_service_times(snapshot, job, picker)
                if policy == "henn" else None),
        ))
        if decision.action == "release":
            planned.append(job)
            selected.extend(decision.jobs)
        else:
            if decision.reconsider_at is not None:
                reconsider.append((decision.reconsider_at, "wait"))
                if decision.reconsider_at > job.start_time:
                    delay = decision.reconsider_at - job.start_time
                    job = replace(job, start_time=decision.reconsider_at,
                                  end_time=job.end_time + delay)
            planned.append(job)

    if commitment.rule == "release_next_per_picker":
        first = {}
        for job in sorted(selected, key=lambda item: (item.start_time,
                                                      item.picker_id)):
            first.setdefault(job.picker_id, job)
        selected = list(first.values())
    elif commitment.rule == "release_within":
        horizon = now + commitment.within_s
        future = [job.start_time - commitment.within_s
                  for job in selected if job.start_time > horizon]
        reconsider.extend((time, "release_window") for time in future)
        selected = [job for job in selected if job.start_time <= horizon]
    return ReleaseDecision(tuple(planned), tuple(selected),
                           tuple(sorted(set(reconsider))))
