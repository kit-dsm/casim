from dataclasses import replace

from ware_ops_algos.algorithms import GreedyItemAssignment, MinMinItemAssignment, RecordedItemAssignment

from casim.pipelines.problem_based_template import AbstractItemAssignment, load_pickle, recorded_plan


class GreedyIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        storage = self._get_storage()
        item_assigner = GreedyItemAssignment(storage)
        return item_assigner


class RecordedIA(AbstractItemAssignment):
    """Keep the recorded pick node for every order position.

    A decision may solve for part of the recorded orders (a pull decision
    plans the orders still open); the plan's lines of the other orders are
    left out, and every line of an order to assign must still be matched.
    """

    def _get_inited_ia(self):
        return _OpenOrdersRecordedIA(self._get_storage(), recorded_plan(self))


class _OpenOrdersRecordedIA(RecordedItemAssignment):
    def _run(self, input_data):
        whole = self.plan
        open_ids = {order.order_id for order in input_data}
        tours = (replace(tour, lines=tuple(line for line in tour.lines
                                           if line.order_id in open_ids))
                 for tour in whole.tours)
        self.plan = replace(whole, tours=tuple(tour for tour in tours if tour.lines))
        try:
            return super()._run(input_data)
        finally:
            self.plan = whole


class MinMinIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        network = load_pickle(self.input()["instance"]["layout"].path).layout_network
        return MinMinItemAssignment(
            storage_locations=self._get_storage(),
            distance_matrix=network.distance_matrix,
            start_node=network.start_node,
        )
