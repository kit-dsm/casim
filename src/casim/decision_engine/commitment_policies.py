from dataclasses import replace

from ware_ops_algos.algorithms import AlgorithmSolution, SchedulingSolution

from casim.domain_objects.sim_domain import SimWarehouseDomain


class CommitmentPolicy:
    def apply(self, solution: AlgorithmSolution, state_snapshot: SimWarehouseDomain) -> AlgorithmSolution:
        raise NotImplementedError


class SchedulingCommitmentPolicy(CommitmentPolicy):
    """Select a deterministic executable prefix from a full schedule.

    This policy controls commitment only.  The StateAdapter controls which
    orders, batches, and resources the solver sees.
    """

    def __init__(
        self,
        n_jobs: int | None = None,
        max_jobs_per_picker: int | None = None,
        planning_horizon_s: float | None = None,
    ):
        self.n_jobs = n_jobs
        self.max_jobs_per_picker = max_jobs_per_picker
        self.planning_horizon_s = planning_horizon_s

    def apply(self, solution: SchedulingSolution, state_snapshot: SimWarehouseDomain) -> SchedulingSolution:
        committed = sorted(
            solution.jobs,
            key=lambda job: (
                job.start_time,
                job.end_time,
                int(job.picker_id),
                job.job.job_id,
            ),
        )
        if self.planning_horizon_s is not None:
            now = float(state_snapshot.dynamic_warehouse_info.time or 0.0)
            limit = now + float(self.planning_horizon_s)
            committed = [job for job in committed if job.start_time <= limit]
        if self.max_jobs_per_picker is not None:
            per_picker: dict[int, int] = {}
            selected = []
            for job in committed:
                picker_id = int(job.picker_id)
                count = per_picker.get(picker_id, 0)
                if count >= self.max_jobs_per_picker:
                    continue
                selected.append(job)
                per_picker[picker_id] = count + 1
            committed = selected
        if self.n_jobs is not None:
            committed = committed[: self.n_jobs]
        return replace(solution, jobs=committed)


class CommitAllPolicy(CommitmentPolicy):
    def apply(self, solution: AlgorithmSolution, state_snapshot: SimWarehouseDomain) -> AlgorithmSolution:
        return solution
