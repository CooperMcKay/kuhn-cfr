"""Fixed opponent strategies, in the same infoset -> (P(pass), P(bet)) format."""

from .game import INFOSETS

PASS = (1.0, 0.0)
BET = (0.0, 1.0)
MIX = (0.5, 0.5)


def _facing_bet(infoset: str) -> bool:
    return infoset.endswith("b")


def always_bet():
    """Bets when opening, calls when facing a bet."""
    return {key: BET for key in INFOSETS}


def always_call():
    """Never bets on its own; calls every bet."""
    return {key: BET if _facing_bet(key) else PASS for key in INFOSETS}


def random_play():
    return {key: MIX for key in INFOSETS}


def heuristic():
    """Value-bets K, calls with Q or K, never bluffs, folds J to a bet."""
    strategy = {}
    for key in INFOSETS:
        card = key[0]
        if _facing_bet(key):
            strategy[key] = BET if card in "QK" else PASS
        else:
            strategy[key] = BET if card == "K" else PASS
    return strategy


BASELINES = {
    "always-call": always_call,
    "always-bet": always_bet,
    "random": random_play,
    "heuristic": heuristic,
}
