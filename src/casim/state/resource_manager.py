from collections import defaultdict

from ware_ops_algos.algorithms import RouteNode
from ware_ops_algos.domain_models import Resources, Resource, CobotPicker
from ware_ops_pipes.utils.experiment_utils import Node # TODO fix where this comes from


class ResourceManager:
    def __init__(self, resources: Resources):
        self._resources = resources
        self._busy_until = {r.id: 0.0 for r in resources.resources}
        self._aisle_count: dict[int, int] = defaultdict(int)  # aisle_id -> picker count
        self._picker_current_aisle: dict[int, int | None] = {}  # picker_id -> aisle_id

    def get_resources(self) -> Resources:
        return self._resources

    def get_resource(self, picker_id: int) -> Resource:
        return self._resources.resources[picker_id]

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
        if isinstance(res, CobotPicker):
            assert isinstance(node.position, tuple)
            self._update_aisle_occupancy(picker_id, node.position)

    def picker_busy_until(self, picker_id: int) -> float:
        return self._busy_until[picker_id]

    def set_picker_busy_until(self, picker_id: int, t: float) -> None:
        if t > self._busy_until[picker_id]:
            self._busy_until[picker_id] = float(t)

    def clear_picker_busy_until(self, picker_id: int, ) -> None:
        self._busy_until[picker_id] = float(self.current_time)

    def _get_resource(self, picker_id: int) -> Resource:
        return self._resources.resources[picker_id]

    def get_aisle_count(self, aisle_id: int = None) -> int | dict:
        if aisle_id is not None:
            return self._aisle_count[aisle_id]
        else:
            return self._aisle_count

    def _update_aisle_occupancy(self, picker_id: int, node: Node) -> None:
        new_aisle = self._get_aisle_id(node)
        old_aisle = self._picker_current_aisle.get(picker_id)

        if old_aisle == new_aisle:
            return

        if old_aisle is not None:
            self._aisle_count[old_aisle] = max(0, self._aisle_count[old_aisle] - 1)

        if new_aisle is not None:
            self._aisle_count[new_aisle] += 1

        self._picker_current_aisle[picker_id] = new_aisle
        # print("Aisle congestion", self._aisle_count)

    @staticmethod
    def _get_aisle_id(node: Node) -> int | float | None:
        if node is None:
            return None
        aisle, slot = node
        if slot < 0:
            return None
        return aisle
