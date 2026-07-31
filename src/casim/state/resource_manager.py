import copy

from ware_ops_algos.algorithms import RouteNode
from ware_ops_algos.domain_models import Resources, Resource

class ResourceManager:
    def __init__(self, resources: Resources):
        self._resources = resources
        self._by_id = {resource.id: resource for resource in resources.resources}
        if len(self._by_id) != len(resources.resources):
            raise ValueError("Resource IDs must be unique")
        self._busy_until = {r.id: 0.0 for r in resources.resources}

    def get_resources(self) -> Resources:
        return self._resources

    def planning_snapshot(
        self,
        resource_ids: set[int] | None = None,
    ) -> Resources:
        resources = [
            copy.deepcopy(resource)
            for resource in self._resources.resources
            if resource_ids is None or resource.id in resource_ids
        ]
        return Resources(self._resources.tpe, resources)

    def get_resource(self, picker_id: int) -> Resource:
        try:
            return self._by_id[picker_id]
        except KeyError as exc:
            raise ValueError(f"Unknown picker ID {picker_id}") from exc

    def mark_picker_occupied(self, picker_id: int) -> None:
        res = self._get_resource(picker_id)
        res.occupied = True

    def mark_picker_free(self, picker_id: int) -> None:
        res = self._get_resource(picker_id)
        res.occupied = False

    def update_resource_location(self, picker_id: int, node: RouteNode) -> None:
        assert isinstance(node, RouteNode)
        res = self._get_resource(picker_id)
        res.current_location = node

    def picker_busy_until(self, picker_id: int) -> float:
        return self._busy_until[picker_id]

    def set_picker_busy_until(self, picker_id: int, t: float) -> None:
        if t > self._busy_until[picker_id]:
            self._busy_until[picker_id] = float(t)

    def set_picker_available(self, picker_id: int):
        self._get_resource(picker_id).available = True

    def set_picker_unavailable(self, picker_id: int):
        self._get_resource(picker_id).available = False

    def clear_picker_busy_until(self, picker_id: int, ) -> None:
        self._busy_until[picker_id] = 0.0

    def _get_resource(self, picker_id: int) -> Resource:
        return self.get_resource(picker_id)
