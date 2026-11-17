from collections import defaultdict

from ware_ops_algos.algorithms import RouteNode
from ware_ops_algos.domain_models import Resources, Resource, CobotPicker
from ware_ops_algos.planning.lifecycle import ResourceStates

Node = tuple[float, float]


class ResourceManager:
    """Resources, their presence and their queue versions (warehouse state).

    Presence changes only by ``PickerArrival`` and ``PickerDeparture``; a
    resource is absent until its first arrival. A departure becomes effective
    at the resource's next free point (``Dispatch``). The queue version grows
    with every change of the resource's queue or presence; a ``Dispatch``
    carrying an older version is stale.
    """

    def __init__(self, resources: Resources):
        self._resources = resources
        self._present = {r.id: False for r in resources.resources}
        # picker_id -> requested absence: a duration (a break) or None (a shift end)
        self._departure: dict[int, float | None] = {}
        # picker_id -> the end of its current absence, when it is known (a break)
        self._return_at: dict[int, float] = {}
        self._version = {r.id: 0 for r in resources.resources}
        self._idle = {r.id: False for r in resources.resources}
        self._aisle_count: dict[int, int] = defaultdict(int)  # aisle_id -> picker count
        self._picker_current_aisle: dict[int, int | None] = {}  # picker_id -> aisle_id

    def get_resources(self) -> Resources:
        return self._resources

    def get_resource(self, picker_id: int) -> Resource:
        return self._resources.resources[picker_id]

    def resource_status(self, picker_id: int, tour_manager) -> ResourceStates:
        """The derived state: busy before off shift, then reserved, then idle."""
        if tour_manager.current_tour(picker_id) is not None:
            return ResourceStates.BUSY
        if not self._present[picker_id]:
            return ResourceStates.OFF_SHIFT
        if tour_manager.queue(picker_id):
            return ResourceStates.RESERVED
        return ResourceStates.IDLE

    def is_present(self, picker_id: int) -> bool:
        return self._present[picker_id]

    def arrive(self, picker_id: int) -> int:
        """Make the resource present; cancels a pending departure."""
        self._present[picker_id] = True
        self._departure.pop(picker_id, None)
        self._return_at.pop(picker_id, None)
        return self.bump_version(picker_id)

    def request_departure(self, picker_id: int, duration: float | None) -> int:
        self._departure[picker_id] = duration
        return self.bump_version(picker_id)

    def has_pending_departure(self, picker_id: int) -> bool:
        return picker_id in self._departure

    def make_departure_effective(self, picker_id: int, now: float) -> float | None:
        """Make the resource absent; return the duration of its absence, if known."""
        duration = self._departure.pop(picker_id)
        if duration is not None:
            self._return_at[picker_id] = now + duration
        self._present[picker_id] = False
        self._idle[picker_id] = False
        self.bump_version(picker_id)
        return duration

    def free_at(self, picker_id: int, now: float, tour_manager) -> float | None:
        """When the resource can begin its next tour; None if absent without known return."""
        current = tour_manager.current_tour(picker_id)
        free = now if current is None else max(
            now, tour_manager.get_tour(current).planned_end)
        if picker_id in self._departure:
            duration = self._departure[picker_id]
            return None if duration is None else free + duration
        if not self._present[picker_id]:
            return self._return_at.get(picker_id)
        return free

    def version(self, picker_id: int) -> int:
        return self._version[picker_id]

    def bump_version(self, picker_id: int) -> int:
        self._version[picker_id] += 1
        return self._version[picker_id]

    def enter_idle(self, picker_id: int) -> bool:
        """Record the resource as idle; True on a transition into idle."""
        entered = not self._idle[picker_id]
        self._idle[picker_id] = True
        return entered

    def leave_idle(self, picker_id: int) -> None:
        self._idle[picker_id] = False

    def update_resource_location(self, picker_id: int, node: RouteNode) -> None:
        assert isinstance(node, RouteNode)
        res = self._get_resource(picker_id)
        res.current_location = node
        if isinstance(res, CobotPicker):
            assert isinstance(node.position, tuple)
            self._update_aisle_occupancy(picker_id, node.position)

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

    @staticmethod
    def _get_aisle_id(node: Node) -> int | float | None:
        if node is None:
            return None
        aisle, slot = node
        if slot < 0:
            return None
        return aisle
