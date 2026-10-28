"""Vanilla counterfactual regret minimization for Kuhn poker."""

import numpy as np

from .game import ACTIONS, DEALS, INFOSETS, acting_player, infoset_key, is_terminal, payoff

Strategy = dict[str, tuple[float, float]]


class _Node:
    """Accumulated regrets and strategy sums for one information set."""

    def __init__(self) -> None:
        self.regret_sum = np.zeros(len(ACTIONS))
        self.strategy_sum = np.zeros(len(ACTIONS))

    def current_strategy(self, reach_weight: float) -> np.ndarray:
        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()
        strategy = positive / total if total > 0 else np.full(len(ACTIONS), 1 / len(ACTIONS))
        self.strategy_sum += reach_weight * strategy
        return strategy

    def average_strategy(self) -> tuple[float, float]:
        total = self.strategy_sum.sum()
        avg = self.strategy_sum / total if total > 0 else np.full(len(ACTIONS), 1 / len(ACTIONS))
        return tuple(float(p) for p in avg)


def _cfr(nodes, cards, history, reach):
    """Return expected utility for player 0; reach = (p0 reach, p1 reach)."""
    if is_terminal(history):
        return float(payoff(cards, history))

    player = acting_player(history)
    node = nodes[infoset_key(cards, history, player)]
    strategy = node.current_strategy(reach[player])

    action_utils = np.zeros(len(ACTIONS))
    for i, action in enumerate(ACTIONS):
        next_reach = tuple(
            reach[p] * strategy[i] if p == player else reach[p] for p in (0, 1)
        )
        action_utils[i] = _cfr(nodes, cards, history + action, next_reach)

    node_util = float(strategy @ action_utils)
    # Regrets are from the acting player's perspective; utilities are player 0's.
    sign = 1.0 if player == 0 else -1.0
    opp_reach = reach[1 - player]
    node.regret_sum += opp_reach * sign * (action_utils - node_util)
    return node_util


def train(iterations: int) -> Strategy:
    """Run `iterations` sweeps over all 6 deals and return the average strategy.

    Maps each infoset key to (P(pass), P(bet)).
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    nodes = {key: _Node() for key in INFOSETS}
    for _ in range(iterations):
        for cards in DEALS:
            _cfr(nodes, cards, "", (1.0, 1.0))
    return {key: node.average_strategy() for key, node in nodes.items()}
