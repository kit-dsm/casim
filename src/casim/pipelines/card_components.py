"""Match algorithm cards to concrete CoSy components through their base classes."""

from ware_ops_algos.algorithms.algorithm_cards import AlgorithmCard


def card_matches_component(card: AlgorithmCard, component: type) -> bool:
    name = card.implementation.get("component_name", card.algo_name)
    return any(base.__name__ == name for base in component.__mro__)
