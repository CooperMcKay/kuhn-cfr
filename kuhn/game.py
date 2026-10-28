"""Kuhn poker rules: deals, histories, information sets, payoffs.

Cards are 0=J, 1=Q, 2=K. Actions are 'p' (pass/check/fold) and 'b' (bet/call).
A history is the string of actions so far; player 0 acts on even-length
histories. Payoffs are always from player 0's (Player 1's) point of view.
"""

from itertools import permutations

CARD_NAMES = ("J", "Q", "K")
ACTIONS = ("p", "b")
DEALS = tuple(permutations(range(3), 2))  # (player 0 card, player 1 card)
TERMINAL_HISTORIES = frozenset({"pp", "bp", "bb", "pbp", "pbb"})
NONTERMINAL_HISTORIES = ("", "p", "b", "pb")

INFOSETS = tuple(
    f"{card}{history}" for card in CARD_NAMES for history in NONTERMINAL_HISTORIES
)


def is_terminal(history: str) -> bool:
    return history in TERMINAL_HISTORIES


def acting_player(history: str) -> int:
    return len(history) % 2


def infoset_key(cards: tuple[int, int], history: str, player: int) -> str:
    """What `player` knows: their own card and the public action history."""
    return f"{CARD_NAMES[cards[player]]}{history}"


def payoff(cards: tuple[int, int], history: str) -> int:
    """Net chips won by player 0 at a terminal history."""
    if not is_terminal(history):
        raise ValueError(f"history {history!r} is not terminal")
    showdown_winner_sign = 1 if cards[0] > cards[1] else -1
    if history == "pp":
        return showdown_winner_sign
    if history == "bp":  # player 1 folded to the bet
        return 1
    if history == "pbp":  # player 0 folded to the bet
        return -1
    return 2 * showdown_winner_sign  # "bb" or "pbb": called bet, pot of 2 each
