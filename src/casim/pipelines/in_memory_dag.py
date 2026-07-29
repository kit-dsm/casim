class InMemoryDagExecutor:
    def __init__(self):
        self.executed: set[str] = set()

    def execute(self, task):
        task_key = self._task_key(task)

        if task_key in self.executed:
            return

        reqs = task.requires()

        if isinstance(reqs, dict):
            for dep in reqs.values():
                self.execute(dep)
        elif isinstance(reqs, (list, tuple)):
            for dep in reqs:
                self.execute(dep)
        elif reqs is not None:
            self.execute(reqs)

        if not self._outputs_exist(task):
            task.run()

        self.executed.add(task_key)

    def execute_many(self, tasks):
        for task in tasks:
            self.execute(task)

    @staticmethod
    def _task_key(task) -> str:
        return getattr(task, "task_id", repr(task))

    @staticmethod
    def _outputs_exist(task) -> bool:
        out = task.output()

        def exists(x):
            if isinstance(x, dict):
                return all(exists(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return all(exists(v) for v in x)
            return x.exists()

        return exists(out)