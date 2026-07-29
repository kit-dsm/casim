from ware_ops_algos.algorithms.order_splitting.order_splitting import LayerPackingSplitting
from ware_ops_algos.domain_models import PalletSpec

from casim.pipelines.problem_based_template import OrderSplitter, load_pickle


class SimpleSplitter(OrderSplitter):
    def _get_order_splitter(self):
        articles = load_pickle(self.input()["instance"]["articles"].path)
        splitter = LayerPackingSplitting(
            pallet_spec=PalletSpec(length=1200, width=800, max_height=1800),  # TODO Must come from data card
            articles=articles,
        )
        return splitter