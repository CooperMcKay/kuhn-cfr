"""Exact evaluation: expected value, best response, exploitability, head-to-head."""

import random
from itertools import product

from .game import ACTIONS, DEALS, INFOSETS, acting_player, infoset_key, is_terminal, payoff


def _expected_value(cards, history, strategies):
    """Expected payoff to player 0 for one deal under per-player strategies."""
    if is_terminal(history):
        return float(payoff(cards, history))
    player = acting_player(history)
    probs = strategies[player][infoset_key(cards, history, player)]
    return sum(
        p * _expected_value(cards, history + action, strategies)
        for p, action in zip(probs, ACTIONS)
        if p > 0
    )


def game_value(strategy0, strategy1) -> float:
    """Expected chips per hand for player 0 when 0 plays strategy0 and 1 plays strategy1."""
    strategies = (strategy0, strategy1)
    return sum(_expected_value(cards, "", strategies) for cards in DEALS) / len(DEALS)


def _player_infosets(player):
    return tuple(k for k in INFOSETS if acting_player(k[1:]) == player)


def _pure_strategies(player):
    keys = _player_infosets(player)
    for choice in product(range(len(ACTIONS)), repeat=len(keys)):
        yield {k: tuple(1.0 if i == c else 0.0 for i in range(len(ACTIONS))) for k, c in zip(keys, choice)}


def best_response_value(opponent_strategy, player: int) -> float:
    """Best achievable chips/hand for `player` against a fixed opponent.

    Exact: a best response can be taken to be pure, and each player has only
    6 infosets (64 pure strategies), so enumerate them all.
    """
    best = float("-inf")
    for pure in _pure_strategies(player):
        if player == 0:
            value = game_value(pure, opponent_strategy)
        else:
            value = -game_value(opponent_strategy, pure)
        best = max(best, value)
    return best


def exploitability(strategy) -> float:
    """Average chips/hand a best-responding opponent gains over the game value.

    Zero iff `strategy` (played by both seats) is a Nash equilibrium.
    """
    return (best_response_value(strategy, 0) + best_response_value(strategy, 1)) / 2


def head_to_head(strategy, opponent) -> float:
    """Exact avg chips/hand for `strategy` vs `opponent`, averaged over both seats."""
    as_p0 = game_value(strategy, opponent)
    as_p1 = -game_value(opponent, strategy)
    return (as_p0 + as_p1) / 2


def _sample_action(probs, rng):
    return ACTIONS[0] if rng.random() < probs[0] else ACTIONS[1]


def simulate_win_rate(strategy, opponent, hands: int, seed: int = 0) -> float:
    """Fraction of sampled hands `strategy` wins chips, alternating seats."""
    rng = random.Random(seed)
    wins = 0
    for n in range(hands):
        cards = rng.choice(DEALS)
        seat = n % 2
        strategies = (strategy, opponent) if seat == 0 else (opponent, strategy)
        history = ""
        while not is_terminal(history):
            player = acting_player(history)
            history += _sample_action(strategies[player][infoset_key(cards, history, player)], rng)
        net = payoff(cards, history)
        wins += (net > 0) if seat == 0 else (net < 0)
    return wins / hands
