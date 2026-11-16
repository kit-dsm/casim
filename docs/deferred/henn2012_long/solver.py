"""Scenario-local card for Henn LONG, leaving CASOP's cards unchanged."""

from pathlib import Path

from ware_ops_algos.algorithms.algorithm_cards import load_algo_card

from casim.pipelines.pipeline_runner import CoSySolver


class HennLongSolver(CoSySolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_cards = [card for card in self.algorithm_cards
                                if card.algo_name != "LPTScheduler"]
        self.algorithm_cards.append(load_algo_card(
            Path(__file__).with_name("henn_long.yaml")))
