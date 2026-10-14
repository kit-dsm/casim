"""The configured wait thresholds share one algorithm card."""

from hydra.utils import get_class
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards

from casim.pipelines.card_components import card_matches_component
from casim.pipelines.subproblems.waiting import NoWaitingNode


def test_order_count_card_matches_each_configured_threshold():
    cards = {card.algo_name: card for card in load_packaged_algo_cards()}
    for threshold in range(1, 5):
        card = cards[f"OrderCountWaiting{threshold}"]
        component = get_class(
            f"casim.pipelines.subproblems.generated.order_count_waiting{threshold}."
            f"OrderCountWaiting{threshold}")
        assert component.min_orders == threshold
        assert card_matches_component(card, component)
        assert not card_matches_component(card, NoWaitingNode)
