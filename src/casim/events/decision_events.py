from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ware_ops_algos.algorithms import BatchObject

from casim.events.base_events import ProcessEvent, Event

if TYPE_CHECKING:
    from casim.state import State

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PickListCreated(ProcessEvent):
    def __init__(self, time: float, pick_list: BatchObject):
        super().__init__(time)
        self.pick_list = pick_list

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        events_to_return = []
        return events_to_return


