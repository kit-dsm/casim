"""The configured wait thresholds share one algorithm card."""

from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards

from casim.pipelines.card_components import card_matches_component
from casim.pipelines.subproblems.waiting import (
    NoWaitingNode, WaitForOne, WaitForTwo, WaitForThree, WaitForFour,
)


def test_order_count_card_matches_each_configured_threshold():
    cards = [card for card in load_packaged_algo_cards()
             if card.algo_name == "OrderCountWaiting"]
    assert len(cards) == 1
    card = cards[0]
    for threshold, component in enumerate(
            (WaitForOne, WaitForTwo, WaitForThree, WaitForFour), start=1):
        assert component.min_orders == threshold
        assert card_matches_component(card, component)
    assert not card_matches_component(card, NoWaitingNode)
