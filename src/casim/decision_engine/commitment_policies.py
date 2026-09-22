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
