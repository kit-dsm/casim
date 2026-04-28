from ware_ops_algos.algorithms import AlgorithmSolution, SchedulingSolution

from casim.domain_objects.sim_domain import SimWarehouseDomain


class CommitmentPolicy:
    def apply(self, solution: AlgorithmSolution, state_snapshot: SimWarehouseDomain) -> AlgorithmSolution:
        raise NotImplementedError


class SchedulingCommitmentPolicy(CommitmentPolicy):
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def apply(self, solution: SchedulingSolution, state_snapshot: SimWarehouseDomain) -> SchedulingSolution:
        committed = sorted(solution.jobs, key=lambda j: j.start_time)[:self.n_jobs]
        return SchedulingSolution(jobs=committed, execution_time=solution.execution_time)


class CommitAllPolicy(CommitmentPolicy):
    def apply(self, solution: AlgorithmSolution, state_snapshot: SimWarehouseDomain) -> AlgorithmSolution:
        return solution


class HennWaiting(CommitmentPolicy):
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def apply(self, solution: SchedulingSolution, state_snapshot) -> SchedulingSolution:
        # Preserve Hs order from the sequencer.
        jobs = solution.jobs

        if not jobs:
            return SchedulingSolution(
                jobs=[],
                execution_time=solution.execution_time,
            )

        job = jobs[0]

        # Henn: if multiple batches exist, select one by Hs and release it.
        if len(jobs) > 1:
            return SchedulingSolution(
                jobs=[job],
                execution_time=solution.execution_time,
            )

        # Henn: if the last order is known, do not postpone.
        if state_snapshot.dynamic_warehouse_info.done:
            return SchedulingSolution(
                jobs=[job],
                execution_time=solution.execution_time,
            )

        # Henn: single-batch case.
        pl = job.route.pick_list

        if pl.service_time is None:
            raise ValueError("Henn waiting requires PickList.service_time.")

        if not pl.single_order_service_times:
            raise ValueError(
                "Henn waiting requires PickList.single_order_service_times."
            )

        orders_by_id = {o.order_id: o for o in pl.orders}

        i_star, st_i = max(
            pl.single_order_service_times.items(),
            key=lambda item: item[1],
        )

        if i_star not in orders_by_id:
            raise ValueError(
                f"Order {i_star} has a service time but is not in PickList.orders."
            )

        r_i = orders_by_id[i_star].order_date
        st_j = pl.service_time

        henn_release_time = 2 * r_i + st_i - st_j
        current_time = state_snapshot.dynamic_warehouse_info.time

        if current_time < henn_release_time:
            duration = job.end_time - job.start_time

            job.start_time = henn_release_time
            job.end_time = henn_release_time + duration
            job.release_time = henn_release_time
            pl.release = henn_release_time

        return SchedulingSolution(
            jobs=[job],
            execution_time=solution.execution_time,
        )
